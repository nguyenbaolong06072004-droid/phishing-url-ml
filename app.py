
import io, os, re, json, time, html, base64, threading, csv, zipfile, tempfile, math, ipaddress
from datetime import datetime
from urllib.parse import urlsplit
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, request, redirect
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from xgboost import XGBClassifier

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1GB

DATASET_DIR, MODEL_DIR, HISTORY_FILE = "saved_datasets", "saved_models", "history.csv"
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DATASETS, CURRENT = {}, {"name": None}
URL_BUNDLES = {}  # ten_dataset -> {"models": {ten: (kind, model)}, "accs": {ten: acc}}
JOB = {"status": "idle", "done": 0, "total": 0, "current": "", "error": "", "body": ""}
BASE_MODELS = ["Logistic Regression", "Decision Tree", "SVM", "Random Forest", "XGBoost"]
SVM_RBF_LIMIT = 20000  # tren muc nay SVM RBF qua cham -> dung SVM tuyen tinh
URL_SAMPLE_LIMIT = 100000
MASTER_NAME = "training_master"
MASTER_FILE = os.path.join(DATASET_DIR, MASTER_NAME + ".csv")


#  Lich su / dataset 
def load_history():
    try:
        return pd.read_csv(HISTORY_FILE).to_dict("records")
    except Exception:
        return []

HISTORY = load_history()

def save_history():
    pd.DataFrame(HISTORY).to_csv(HISTORY_FILE, index=False)

def load_saved_datasets():
    for fname in sorted(os.listdir(DATASET_DIR)):
        if fname.endswith(".csv"):
            try:
                DATASETS[fname[:-4]] = pd.read_csv(os.path.join(DATASET_DIR, fname))
            except Exception:
                pass
    if DATASETS:
        CURRENT["name"] = next(iter(DATASETS))

load_saved_datasets()

META_FILE = os.path.join(DATASET_DIR, "meta.json")
try:
    META = json.load(open(META_FILE, encoding="utf-8"))
except Exception:
    META = {}  # ten_dataset -> {"phishing_value": str, "confirmed": bool}

def save_meta():
    json.dump(META, open(META_FILE, "w", encoding="utf-8"), ensure_ascii=False)

def guess_phishing_value(s):
    """Doan tam gia tri phishing khi nguoi dung khong nhap (se hoi xac nhan tren trang chu)."""
    vals = set(s.dropna().unique())
    if pd.api.types.is_numeric_dtype(s):
        return "-1" if -1 in vals else "1"
    return str(sorted(map(str, vals))[0])


def esc(x):
    return html.escape(str(x))

def make_models(n_samples):
    if n_samples <= SVM_RBF_LIMIT:
        svm = ("SVM (RBF)", SVC(kernel="rbf", random_state=42))
    else:
        svm = ("SVM (Linear)", CalibratedClassifierCV(LinearSVC(dual=False, random_state=42), cv=3))
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        svm[0]: svm[1],
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42, tree_method="hist", n_jobs=-1),
    }

def _read_text_sample(fs, size=65536):
    pos = fs.tell()
    fs.seek(0)
    raw = fs.read(size)
    fs.seek(pos)
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8"


def _looks_like_url(value):
    s = str(value).strip().lower()
    return bool(re.search(r"(?:https?|ftp)://|^www\\.|^[a-z0-9.-]+\\.[a-z]{2,}(?:/|$)", s))


def _detect_text_separator(sample):
    lines = [x.strip() for x in sample.splitlines() if x.strip()][:100]
    if not lines:
        return None
    # TXT dataset phishing pho bien nhat la label<TAB>URL -> uu tien TAB.
    for sep in ("\t", ",", ";", "|"):
        counts = [line.count(sep) for line in lines]
        if sum(c > 0 for c in counts) >= max(3, int(len(lines) * 0.6)):
            return sep
    return None


def _load_delimited_text(fs, sep, encoding):
    fs.seek(0)
    # Doc bang pandas thay vi fs.read().splitlines() de dataset hang tram nghin dong khong
    # tao them mot list/string khong lo trong RAM.
    df = pd.read_csv(fs, sep=sep, header=None, dtype=str, keep_default_na=False,
                     encoding=encoding, engine="python", on_bad_lines="skip")
    if df.empty:
        raise ValueError("File TXT khong co du lieu.")
    # Bo cot rong o cuoi neu co.
    df = df.dropna(axis=1, how="all")
    if df.shape[1] == 1:
        return pd.DataFrame({"url": df.iloc[:, 0].astype(str).str.strip()})
    if df.shape[1] >= 2:
        a, b = df.iloc[:, 0], df.iloc[:, 1]
        score_a = a.head(2000).map(_looks_like_url).mean()
        score_b = b.head(2000).map(_looks_like_url).mean()
        if score_a > score_b:
            return pd.DataFrame({"url": a.astype(str).str.strip(), "label": b.astype(str).str.strip()})
        return pd.DataFrame({"label": a.astype(str).str.strip(), "url": b.astype(str).str.strip()})
    return df


def load_text_dataset(fs):
    sample, encoding = _read_text_sample(fs)
    lines = [x.strip() for x in sample.splitlines() if x.strip()]
    if not lines:
        raise ValueError("File TXT rong.")
    sep = _detect_text_separator(sample)
    if sep:
        return _load_delimited_text(fs, sep, encoding)
    fs.seek(0)
    raw = fs.read()
    for enc in (encoding, "utf-8", "cp1252", "latin1"):
        try:
            text = raw.decode(enc) if isinstance(raw, bytes) else raw
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="replace")
    return pd.DataFrame({"url": [x.strip() for x in text.splitlines() if x.strip()]})


def load_uploaded_file(fs):
    filename = secure_filename(fs.filename or "dataset")
    ext = os.path.splitext(filename)[1].lower()
    if not ext:
        raise ValueError("File khong co duoi. Ho tro CSV, TSV, TXT, DATA, LOG, JSON, JSONL, XLSX, XLS, ARFF, PARQUET.")

    if ext == ".csv":
        last = None
        for enc in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
            try:
                fs.seek(0)
                return pd.read_csv(fs, encoding=enc, low_memory=False)
            except Exception as e:
                last = e
        raise ValueError(f"Khong doc duoc CSV: {last}")

    if ext == ".tsv":
        fs.seek(0)
        return pd.read_csv(fs, sep="\t", encoding="utf-8-sig", low_memory=False)

    if ext in (".txt", ".data", ".log"):
        return load_text_dataset(fs)

    if ext == ".json":
        fs.seek(0)
        return pd.read_json(fs)

    if ext in (".jsonl", ".ndjson"):
        fs.seek(0)
        return pd.read_json(fs, lines=True)

    if ext in (".xlsx", ".xls"):
        fs.seek(0)
        return pd.read_excel(fs)

    if ext == ".parquet":
        fs.seek(0)
        return pd.read_parquet(fs)

    if ext == ".arff":
        from scipy.io import arff
        fs.seek(0)
        raw = fs.read()
        text = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else raw
        data, _ = arff.loadarff(io.StringIO(text))
        df = pd.DataFrame(data)
        for c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].apply(lambda x: x.decode() if isinstance(x, bytes) else x)
                try:
                    df[c] = pd.to_numeric(df[c])
                except (ValueError, TypeError):
                    pass
        return df

    raise ValueError(f"Chua ho tro dinh dang {ext}. Ho tro CSV, TSV, TXT, DATA, LOG, JSON, JSONL, XLSX, XLS, ARFF, PARQUET.")

def normalize_label(s, phishing_value):
    """Tra ve Series 0/1 voi phishing = 1. Khong doan khi nhan la so."""
    uniq = sorted(map(str, s.dropna().unique()))[:10]
    if phishing_value != "":
        num = pd.to_numeric(s, errors="coerce")
        try:
            return (num == float(phishing_value)).astype(int)
        except ValueError:
            return (s.astype(str).str.strip().str.lower() == phishing_value.lower()).astype(int)
    if not pd.api.types.is_numeric_dtype(s):
        out = s.astype(str).str.lower().str.contains("phish").astype(int)
        if 0 < out.sum() < len(out):
            return out
    raise ValueError(f"Hay nhap gia tri PHISHING. Cac gia tri nhan hien co: {', '.join(uniq)}")

def find_url_column(df):
    names = {str(c).strip().lower(): c for c in df.columns}
    for k in ["url", "urls", "website", "web", "link", "uri", "address", "domain"]:
        if k in names:
            return names[k]
    best, best_score = None, 0
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        sample = df[c].dropna().astype(str).head(2000)
        if len(sample) == 0:
            continue
        score = sample.map(_looks_like_url).mean()
        if score > best_score:
            best, best_score = c, score
    return best if best_score >= 0.25 else None


FEATURE_VERSION = 4  # doi bo dac trung URL -> khong dung mo hinh cu

def normalize_url(u):
    """Bo http://, https://, www. va dau / cuoi de cac cach viet cua cung 1 URL cho cung ket qua."""
    u = re.sub(r"^[a-z][a-z0-9+.-]*://", "", str(u).strip().lower())
    return re.sub(r"^www\d*\.", "", u).rstrip("/")

def find_label_column(df):
    names = {str(c).strip().lower(): c for c in df.columns}
    for k in ["label", "result", "class", "status", "phishing", "target", "type", "y", "category"]:
        if k in names:
            return names[k]
    # Tim cot nhi phan co ten/du lieu giong nhan phan loai.
    candidates = []
    for c in df.columns:
        if str(c).lower() in {"url", "urls", "website", "web", "link", "uri", "domain", "id"}:
            continue
        vals = df[c].dropna().astype(str).str.strip().str.lower()
        uniq = set(vals.unique())
        if len(uniq) == 2:
            label_words = sum(any(w in v for w in ("phish", "legit", "benign", "malicious", "safe", "fraud", "spam")) for v in uniq)
            binary_words = uniq.issubset({"0", "1", "-1", "true", "false", "yes", "no"})
            candidates.append((2 if label_words else 1 if binary_words else 0, c))
    return sorted(candidates, key=lambda x: x[0])[-1][1] if candidates else None

def rebuild_master():
    """Tao lai training_master tu cac dataset nguon hien dang ton tai.
    Khong cong ket qua model; luon gop URL + nhan roi train lai tu dau.
    """
    parts = []
    for name, df in DATASETS.items():
        if name == MASTER_NAME:
            continue
        ucol = find_url_column(df)
        if not ucol or "label" not in df.columns:
            continue
        z = pd.DataFrame({
            "url": df[ucol].astype(str),
            "label": pd.to_numeric(df["label"], errors="coerce"),
            "source_dataset": name
        }).dropna(subset=["label"])
        z["label"] = z["label"].astype(int)
        z["url"] = z["url"].map(normalize_url)
        z = z[z["url"].str.len() > 0]
        if len(z):
            parts.append(z)

    if parts:
        master = pd.concat(parts, ignore_index=True)
        # Neu cung URL xuat hien o nhieu file, uu tien dataset duoc them/sua sau cung.
        master = master.drop_duplicates(subset=["url"], keep="last").reset_index(drop=True)
    else:
        master = pd.DataFrame(columns=["url", "label", "source_dataset"])

    master.to_csv(MASTER_FILE, index=False)
    DATASETS[MASTER_NAME] = master
    META[MASTER_NAME] = {"phishing_value": "1", "confirmed": True, "accumulative": True}
    CURRENT["name"] = MASTER_NAME
    save_meta()
    return master

def accumulate_dataset(df, source_name=None):
    """Them/cap nhat dataset vao tap tong hop va loai URL trung."""
    ucol = find_url_column(df)
    if not ucol:
        raise ValueError("Dataset khong co cot URL de train tich luy.")
    if source_name and source_name in DATASETS:
        DATASETS[source_name] = df
    return rebuild_master()

def _entropy(text):
    """Shannon entropy cua chuoi; dung de phat hien chuoi URL/hostname bat thuong."""
    text = str(text or "")
    if not text:
        return 0.0
    counts = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(text)
    return float(-sum((c / n) * math.log2(c / n) for c in counts.values()))


def _hostname_is_ip(hostname):
    hostname = str(hostname or "").strip("[]")
    try:
        ipaddress.ip_address(hostname)
        return 1
    except ValueError:
        return 0


def extract_basic_features(url):
    """Trich xuat bo dac trung URL thong nhat cho ca train va check.

    Bam theo nhom lexical trong tai lieu: do dai URL, dau cham, subdomain,
    ky tu dac biet, IP, tu khoa dang ngo, chu so, hostname; bo sung cac
    dac trung lexical co the tinh truc tiep tu URL de dat khoang 20-30 dac trung.
    Host-based/content-based can DNS/WHOIS/truy cap web nen khong tu dong goi
    Internet khi nguoi dung nhap URL.
    """
    raw = str(url or "").strip()
    norm = normalize_url(raw)
    parse_target = raw if re.match(r"^[a-z][a-z0-9+.-]*://", raw, re.I) else "//" + norm
    try:
        p = urlsplit(parse_target)
        hostname = (p.hostname or "").lower()
        path = p.path or ""
        query = p.query or ""
        fragment = p.fragment or ""
        scheme = (p.scheme or "").lower()
        port = p.port
    except Exception:
        hostname = ""
        path = ""
        query = ""
        fragment = ""
        scheme = ""
        port = None

    special_chars = "-_/ ?=&@%#;:+,$~()[]{}!|\\'*\"^`<>"
    special_count = sum(norm.count(ch) for ch in special_chars)
    suspicious_keywords = ["login", "verify", "account", "update", "secure", "confirm"]
    keyword_count = sum(norm.lower().count(k) for k in suspicious_keywords)
    labels = [x for x in hostname.split(".") if x]
    num_subdomains = max(len(labels) - 2, 0) if len(labels) >= 2 else 0
    digits = sum(c.isdigit() for c in norm)
    encoded_count = len(re.findall(r"%[0-9a-fA-F]{2}", norm))
    query_params = 0 if not query else query.count("&") + 1
    path_depth = len([x for x in path.split("/") if x])
    unicode_count = sum(ord(c) > 127 for c in norm)
    has_punycode = int("xn--" in hostname)
    has_double_slash_path = int("//" in path)
    has_ip = _hostname_is_ip(hostname)
    has_https = int(scheme == "https")
    has_http = int(scheme == "http")
    has_port = int(port is not None or bool(re.search(r"^[^/]+:\d+", norm)))

    return {
        # 9 dac trung lexical duoc neu truc tiep trong tai lieu
        "url_length": len(norm),
        "num_dots": norm.count("."),
        "num_subdomains": num_subdomains,
        "num_special_chars": special_count,
        "has_ip": has_ip,
        "keyword_count": keyword_count,
        "num_digits": digits,
        "hostname_length": len(hostname),
        "num_hyphens": norm.count("-"),
        # Bo sung lexical de dat bo 20-30 dac trung va giu train/check dong nhat
        "num_at": norm.count("@"),
        "num_slashes": norm.count("/"),
        "num_question": norm.count("?"),
        "num_equals": norm.count("="),
        "num_percent": norm.count("%"),
        "num_ampersands": norm.count("&"),
        "num_underscores": norm.count("_"),
        "num_colons": norm.count(":"),
        "path_length": len(path),
        "query_length": len(query),
        "fragment_length": len(fragment),
        "digit_ratio": digits / max(len(norm), 1),
        "query_param_count": query_params,
        "path_depth": path_depth,
        "has_https": has_https,
        "has_http": has_http,
        "has_port": has_port,
        "has_punycode": has_punycode,
        "has_encoded_chars": int(encoded_count > 0),
        "encoded_char_count": encoded_count,
        "has_fragment": int(bool(fragment)),
        "has_unicode": int(unicode_count > 0),
        "unicode_char_count": unicode_count,
        "has_double_slash_path": has_double_slash_path,
        "hostname_entropy": _entropy(hostname),
        "url_entropy": _entropy(norm),
    }

def fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)

CSS = """<style>
body{font-family:Segoe UI,Arial,sans-serif;background:#0f172a;color:#e2e8f0;margin:0;padding:30px}
.card{background:#1e293b;border-radius:10px;padding:24px;margin-bottom:20px;max-width:1100px}
h1{color:#60a5fa}h2{color:#93c5fd;border-bottom:1px solid #334155;padding-bottom:8px}
button{background:#2563eb;color:#fff;border:none;padding:10px 18px;border-radius:6px;cursor:pointer;font-size:15px}
button:hover{background:#1d4ed8}button.danger{background:#991b1b}
table{border-collapse:collapse;width:100%;margin-top:10px}
th,td{border:1px solid #334155;padding:8px 12px;text-align:center}th{background:#334155}
.ok{color:#4ade80;font-weight:bold}.bad{color:#f87171;font-weight:bold}.hint{color:#94a3b8;font-size:13px}
img{max-width:100%;border-radius:8px;margin-top:10px;background:#fff;padding:6px}
label{display:block;margin:6px 0}a{color:#60a5fa}
input[type=file],input[type=text],input[type=number],select{padding:8px;border-radius:6px;border:1px solid #334155;background:#0f172a;color:#fff}
.box{background:linear-gradient(135deg,#1e3a8a,#1e293b);border-radius:10px;padding:20px;margin-bottom:20px;text-align:center;max-width:1100px}
.box .big{font-size:36px;font-weight:bold}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:700px){.grid{grid-template-columns:1fr}}
tr.best{background:#14532d}tr.best td{color:#86efac;font-weight:bold}
.bar{background:#334155;border-radius:8px;height:22px;overflow:hidden}.bar div{background:#2563eb;height:100%}
</style>"""

def page(body, refresh=False):
    meta = "<meta http-equiv='refresh' content='2'>" if refresh else ""
    return f"<html><head><meta charset='utf-8'>{meta}<title>Thuc nghiem Phishing URL</title>{CSS}</head><body>{body}</body></html>"

def err_page(msg):
    return page(f"<div class='card'><h1>Loi</h1><p class='bad'>{esc(msg)}</p><a href='/'>&larr; Quay lai</a></div>")



@app.route("/")
def home():
    df = DATASETS.get(CURRENT["name"])
    ds = CURRENT["name"]
    n_feat = (df.shape[1] - 1 - int("label_raw" in df.columns)) if df is not None else 0
    status = (f"<span class='ok'>Dang dung: {esc(ds)} - {df.shape[0]} mau, {n_feat} cot</span>"
              if df is not None else "<span class='bad'>Chua co dataset nao</span>")
    switch = ""
    if DATASETS:
        opts = "".join(f"<option value='{esc(n)}' {'selected' if n == ds else ''}>{esc(n)} ({d.shape[0]} mau)</option>"
                       for n, d in DATASETS.items())
        switch = (f"<form method='post' action='/select_dataset'><label>Chon dataset da tai:"
                  f"<select name='dataset_name' onchange='this.form.submit()'>{opts}</select></label></form>")
        if ds and ds != MASTER_NAME:
            switch += (f"<form method='post' action='/delete_dataset' style='margin-top:8px' "
                       f"onsubmit=\"return confirm('Xoa dataset {esc(ds)} va train lai tu cac dataset con lai?')\">"
                       f"<input type='hidden' name='dataset_name' value='{esc(ds)}'>"
                       f"<button type='submit'>Xoa dataset dang chon</button></form>")

    banner = ""
    if df is not None and "label_raw" in df.columns:
        m = META.get(ds, {})
        ucol = find_url_column(df)
        raw = df["label_raw"].astype(str)
        vals = sorted(raw.unique())[:10]
        info = ""
        for v in vals:
            smp = ""
            if ucol:
                smp = "<br>".join(esc(str(u)[:80]) for u in df.loc[raw == v, ucol].head(3))
            info += f"<tr><td>{esc(v)}</td><td>{(raw == v).sum()}</td><td style='text-align:left'>{smp or '-'}</td></tr>"
        cur = str(raw[df["label"] == 1].mode().iloc[0]) if (df["label"] == 1).any() else ""  # gia tri goc dang la phishing
        opts = "".join(f"<option value='{esc(v)}' {'selected' if v == cur else ''}>{esc(v)}</option>" for v in vals)
        warn = "" if m.get("confirmed") else "<p class='bad'>Chua xac nhan: app dang tam doan. Xem URL mau roi bam Ap dung de chay thuc nghiem.</p>"
        banner = f"""<div style="margin-top:12px">{warn}
        <table><tr><th>Gia tri nhan goc</th><th>So mau</th><th>Vi du</th></tr>{info}</table>
        <form method="post" action="/set_phishing"><label>Gia tri nao la PHISHING?
        <select name="phishing_value">{opts}</select> <button type="submit">Ap dung</button></label></form></div>"""

    results = ""
    if df is not None:
        ucol = find_url_column(df)
        links = ""
        if JOB["status"] == "running":
            links += "<a href='/progress'>Dang chay... xem tien do &rarr;</a><br>"
        elif JOB["body"]:
            links += "<a href='/run/result'>Xem ket qua thuc nghiem gan nhat &rarr;</a><br>"
        if ucol and get_bundle(ds):
            links += "<a href='/check_url'>Kiem tra URL &rarr;</a><br>"
        results = (f"<div class='card'><h2>2. Ket qua</h2>{links or '<p>Chua co ket qua.</p>'}"
                   "<form method='post' action='/run'><button type='submit'>Chay lai thuc nghiem</button></form></div>")

    body = f"""<h1>Thuc nghiem danh gia mo hinh hoc may - Phishing URL</h1>
    <div class="card"><h2>1. Dataset</h2><p>{status}</p>{switch}{banner}<hr style="border-color:#334155">
    <p>Tai len dataset MOI (CSV/TSV/TXT/JSON/Excel/ARFF/Parquet):</p>
    <form method="post" action="/upload" enctype="multipart/form-data">
      <input type="file" name="file" accept=".csv,.tsv,.txt,.data,.log,.json,.jsonl,.xlsx,.xls,.arff,.parquet" required><button type="submit">Tai len</button>
    </form></div>
    {results}
    <div class="card"><a href="/history">Xem lich su thuc nghiem ({len(HISTORY)} ket qua) &rarr;</a></div>"""
    return page(body)

@app.route("/select_dataset", methods=["POST"])
def select_dataset():
    name = request.form.get("dataset_name")
    if name in DATASETS:
        CURRENT["name"] = name
    return redirect("/")

@app.route("/set_phishing", methods=["POST"])
def set_phishing():
    ds, df = CURRENT["name"], DATASETS.get(CURRENT["name"])
    v = request.form.get("phishing_value", "")
    if df is None or "label_raw" not in df.columns:
        return redirect("/")
    try:
        lab = normalize_label(df["label_raw"], v)
        if lab.nunique() < 2:
            raise ValueError("Gia tri khong khop nhan nao.")
    except Exception as e:
        return err_page(e)
    df["label"] = lab
    df.to_csv(os.path.join(DATASET_DIR, f"{ds}.csv"), index=False)
    META[ds] = {"phishing_value": v, "confirmed": True}
    save_meta()
    URL_BUNDLES.pop(ds, None)
    if os.path.exists(bundle_path(ds)):
        os.remove(bundle_path(ds))
    master = rebuild_master()
    if master["label"].nunique() < 2:
        return err_page("Sau khi sua nhan, dataset tong hop phai co ca 0 va 1.")
    return redirect("/progress" if start_auto(MASTER_NAME) else "/")

@app.route("/delete_dataset", methods=["POST"])
def delete_dataset():
    ds = request.form.get("dataset_name", "").strip()
    if not ds or ds == MASTER_NAME:
        return err_page("Khong cho phep xoa training_master truc tiep. Hay xoa dataset nguon.")
    if ds not in DATASETS:
        return err_page("Khong tim thay dataset can xoa.")

    DATASETS.pop(ds, None)
    META.pop(ds, None)
    URL_BUNDLES.pop(ds, None)
    path = os.path.join(DATASET_DIR, f"{ds}.csv")
    if os.path.exists(path):
        os.remove(path)
    bp = bundle_path(ds)
    if os.path.exists(bp):
        os.remove(bp)
    save_meta()

    master = rebuild_master()
    URL_BUNDLES.pop(MASTER_NAME, None)
    mbp = bundle_path(MASTER_NAME)
    if os.path.exists(mbp):
        os.remove(mbp)
    if master["label"].nunique() >= 2:
        return redirect("/progress" if start_auto(MASTER_NAME) else "/")
    return redirect("/")

NEED_CONFIRM = "Hay xac nhan gia tri PHISHING o muc 1 (trang chu) truoc."

@app.route("/upload", methods=["POST"])
def upload():
    try:
        fs = request.files["file"]
        pv = request.form.get("phishing_value", "").strip()
        df = load_uploaded_file(fs)
        label_col = find_label_column(df)
        if label_col is None:
            raise ValueError("Khong tu tim duoc cot nhan trong file nay.")
        if label_col != "label":
            df = df.rename(columns={label_col: "label"})
        raw = df["label"].copy()
        try:
            df["label"] = normalize_label(raw, pv)
            confirmed, pv_used = True, (pv or "phish")
        except ValueError:
            pv_used = guess_phishing_value(raw)
            df["label"] = normalize_label(raw, pv_used)
            confirmed = False
        if df["label"].nunique() < 2:
            raise ValueError("Gia tri PHISHING khong khop nhan nao trong du lieu.")
        df["label_raw"] = raw
        base = secure_filename(os.path.splitext(fs.filename)[0]) or "dataset"
        name, k = base, 1
        while name in DATASETS:
            k += 1; name = f"{base}_{k}"
        DATASETS[name] = df
        CURRENT["name"] = name
        df.to_csv(os.path.join(DATASET_DIR, f"{name}.csv"), index=False)
        META[name] = {"phishing_value": pv_used, "confirmed": confirmed}
        save_meta()
        if confirmed:
            master = rebuild_master()
            if master["label"].nunique() < 2:
                raise ValueError("Dataset tong hop chua co du 2 lop Hop le/Phishing.")
            started = start_auto(MASTER_NAME)
            return redirect("/progress" if started else "/")
        return redirect("/")
    except Exception as e:
        return err_page(e)


# ---------------- Thuc nghiem (chay nen) ----------------
def run_job(ds_name, df_full, selected, mode, max_samples, drop_cols, feat_src="dataset", final=True):
    try:
        if feat_src == "url":
            col = find_url_column(df_full)
            if col is None:
                raise ValueError("Dataset khong co cot URL de trich dac trung.")
            d = df_full[[col, "label"]].dropna()
            if len(d) > (max_samples or 200000):
                d = d.sample(max_samples or 200000, random_state=42)
            JOB["current"] = "Trich dac trung tu URL..."
            X = pd.DataFrame([extract_basic_features(u) for u in d[col]])
            y = d["label"].astype(int).to_numpy()
            X = X.replace([np.inf, -np.inf], np.nan)
        else:
            work = df_full.drop(columns=drop_cols + ["label_raw"], errors="ignore").copy()
            # Nhan dien cac cot dang chuoi nhung thuc chat la so.
            for c in work.columns:
                if c == "label" or pd.api.types.is_numeric_dtype(work[c]):
                    continue
                converted = pd.to_numeric(work[c], errors="coerce")
                if len(work) and converted.notna().mean() >= 0.90:
                    work[c] = converted
            num = work.select_dtypes(include=["number"])
            y = num["label"].astype(int).to_numpy()
            X = num.drop(columns=["label", "id", "index"], errors="ignore").replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        if X.shape[1] == 0:
            raise ValueError("Dataset khong co cot dac trung dang so.")
        if max_samples and len(X) > max_samples:
            X, _, y, _ = train_test_split(X, y, train_size=max_samples, stratify=y, random_state=42)
            X = X.reset_index(drop=True)
        models = {n: m for n, m in make_models(len(X)).items() if any(n.startswith(s) for s in selected)}
        JOB.update(total=len(models), done=0)
        is_cv = mode == "cv5"
        if not is_cv:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        else:
            skf = StratifiedKFold(5, shuffle=True, random_state=42)
        rows, fitted, stds = [], {}, {}
        run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for name, model in models.items():
            JOB["current"] = name
            pipe = make_pipeline(StandardScaler(), model)
            if is_cv:
                r = cross_validate(pipe, X, y, cv=skf, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
                row = {"Model": name, "Accuracy": r["test_accuracy"].mean(), "Precision": r["test_precision"].mean(),
                       "Recall": r["test_recall"].mean(), "F1": r["test_f1"].mean(),
                       "ROC_AUC": r["test_roc_auc"].mean(), "Train_s": r["fit_time"].mean()}
                stds[name] = {"Accuracy": r["test_accuracy"].std(), "Precision": r["test_precision"].std(),
                              "Recall": r["test_recall"].std(), "F1": r["test_f1"].std(), "ROC_AUC": r["test_roc_auc"].std()}
            else:
                t0 = time.perf_counter()
                pipe.fit(Xtr, ytr)
                fit_t = time.perf_counter() - t0
                pred = pipe.predict(Xte)
                row = {"Model": name, "Accuracy": accuracy_score(yte, pred),
                       "Precision": precision_score(yte, pred, zero_division=0),
                       "Recall": recall_score(yte, pred, zero_division=0),
                       "F1": f1_score(yte, pred, zero_division=0),
                       "ROC_AUC": roc_auc_score(yte, get_scores(pipe, Xte)), "Train_s": fit_t}
                fitted[name] = pipe
            rows.append(row)
            HISTORY.append({"Thoi_gian": run_time, "Dataset": ds_name, "Che_do": "CV5" if is_cv else "Holdout",
                            "So_mau": len(X), "So_dac_trung": X.shape[1], **{k: v for k, v in row.items()}})
            JOB["done"] += 1
        save_history()

        if feat_src == "url":  # luu chinh cac mo hinh vua thuc nghiem de dung kiem tra URL
            mods, accs = {}, {}
            for r in rows:
                n = r["Model"]
                p = fitted[n] if n in fitted else make_pipeline(StandardScaler(), models[n]).fit(X, y)
                mods[f"{n} (thuc nghiem)"] = ("feat", p)
                accs[f"{n} (thuc nghiem)"] = r["Accuracy"]
            bundle = {"version": FEATURE_VERSION, "models": mods, "accs": accs, "f1s": {f"{r['Model']} (thuc nghiem)": r["F1"] for r in rows}}
            bundle["best"] = max(bundle["f1s"], key=bundle["f1s"].get)
            URL_BUNDLES[ds_name] = bundle
            joblib.dump(bundle, bundle_path(ds_name))

        rows.sort(key=lambda r: r["F1"], reverse=True)
        tr = ""
        for i, r in enumerate(rows):
            cells = ""
            for k in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]:
                s = f"{r[k]*100:.2f}%"
                if is_cv:
                    s += f" &plusmn; {stds[r['Model']][k]*100:.2f}"
                cells += f"<td>{s}</td>"
            tr += f"<tr class='{'best' if i == 0 else ''}'><td>{esc(r['Model'])}</td>{cells}<td>{r['Train_s']:.2f}</td></tr>"
        table = ("<table><tr><th>Mo hinh</th><th>Accuracy</th><th>Precision (phishing)</th><th>Recall (phishing)</th>"
                 f"<th>F1 (phishing)</th><th>ROC-AUC</th><th>Train (s{' /fold' if is_cv else ''})</th></tr>{tr}</table>")

        mdf = pd.DataFrame(rows).set_index("Model")
        fig, ax = plt.subplots(figsize=(7, 4))
        mdf[["F1", "ROC_AUC"]].plot(kind="bar", ax=ax)
        ax.set_title("So sanh hieu nang"); ax.tick_params(axis="x", rotation=20)
        cmp_img = fig_b64(fig)

        extra = ""
        best = rows[0]["Model"]
        if not is_cv:
            cm = confusion_matrix(yte, fitted[best].predict(Xte))
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            ax2.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    ax2.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
            ax2.set_xticks([0, 1]); ax2.set_xticklabels(["Hop le", "Phishing"])
            ax2.set_yticks([0, 1]); ax2.set_yticklabels(["Hop le", "Phishing"])
            ax2.set_xlabel("Du doan"); ax2.set_ylabel("Thuc te"); ax2.set_title(f"Ma tran nham lan - {best}")
            extra += f"<div><p style='text-align:center'>Ma tran nham lan</p><img src='data:image/png;base64,{fig_b64(fig2)}'></div>"
            if "Random Forest" in fitted:
                imp = pd.Series(fitted["Random Forest"][-1].feature_importances_, index=X.columns).nlargest(10)
                fig3, ax3 = plt.subplots(figsize=(6, 4))
                imp.plot(kind="barh", ax=ax3); ax3.invert_yaxis(); ax3.set_title("Top 10 dac trung (Random Forest)")
                fig3.tight_layout()
                extra += f"<div><p style='text-align:center'>Dac trung quan trong</p><img src='data:image/png;base64,{fig_b64(fig3)}'></div>"

        JOB["body"] = f"""<h1>Ket qua thuc nghiem</h1>
        <p class="hint">Dataset: {esc(ds_name)} | {len(X)} mau, {X.shape[1]} dac trung | Che do: {'5-fold CV' if is_cv else 'Holdout 80/20'}
        | Dac trung: {'trich tu URL (da luu de kiem tra URL)' if feat_src == 'url' else 'cot co san'} | Bo cot: {esc(', '.join(drop_cols) or 'khong')}. Metric tinh tren lop phishing (=1).</p>
        <div class="box"><div class="big ok">{esc(best)}</div>
        <div>F1 {rows[0]['F1']*100:.2f}% | ROC-AUC {rows[0]['ROC_AUC']*100:.2f}%</div></div>
        <div class="card"><h2>Bang so sanh</h2>{table}</div>
        <div class="card"><h2>Bieu do</h2><div class="grid"><div><img src="data:image/png;base64,{cmp_img}"></div>{extra}</div></div>
        <p><a href="/">&larr; Trang chu</a> | <a href="/history">Lich su &rarr;</a></p>"""
        if final:
            JOB["status"] = "done"
    except Exception as e:
        JOB.update(status="error", error=str(e))

def add_tfidf_model(ds, df):
    col = find_url_column(df)
    d = df[[col, "label"]].dropna()
    if len(d) > 200000:
        d = d.sample(200000, random_state=42)
    urls, y = d[col].map(normalize_url).reset_index(drop=True), d["label"].astype(int).to_numpy()
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
    t0 = time.perf_counter()
    txt = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=200000, sublinear_tf=True),
                        LogisticRegression(max_iter=1000)).fit(urls.iloc[tr], y[tr])
    fit_t = time.perf_counter() - t0
    pred = txt.predict(urls.iloc[te])
    name = "Logistic Regression (TF-IDF ky tu)"
    b = get_bundle(ds) or {"version": FEATURE_VERSION, "models": {}, "accs": {}, "f1s": {}}
    b["models"][name] = ("text", txt)
    b["accs"][name] = accuracy_score(y[te], pred)
    b.setdefault("f1s", {})[name] = f1_score(y[te], pred, zero_division=0)
    b["best"] = max(b["f1s"], key=b["f1s"].get)
    URL_BUNDLES[ds] = b
    joblib.dump(b, bundle_path(ds))
    HISTORY.append({"Thoi_gian": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Dataset": ds, "Che_do": "Holdout",
                    "So_mau": len(y), "So_dac_trung": "TF-IDF ky tu", "Model": name,
                    "Accuracy": b["accs"][name], "Precision": precision_score(y[te], pred, zero_division=0),
                    "Recall": recall_score(y[te], pred, zero_division=0), "F1": b["f1s"][name],
                    "ROC_AUC": roc_auc_score(y[te], txt.predict_proba(urls.iloc[te])[:, 1]), "Train_s": fit_t})
    save_history()

def auto_job(ds):
    try:
        df = DATASETS[ds]
        ucol = find_url_column(df)
        n_cols = df.select_dtypes(include=["number"]).drop(columns=["label", "label_raw"], errors="ignore").shape[1]
        plans = []
        if n_cols > 0:
            plans.append(("dataset", f"Ket qua 1: dung {n_cols} cot dac trung co san trong dataset"))
        if ucol:
            plans.append(("url", "Ket qua 2: dac trung trich tu chuoi URL (dung de kiem tra URL)"))
        if not plans:
            raise ValueError("Dataset khong co cot dac trung dang so va cung khong co cot URL.")
        parts = []
        for src, title in plans:
            run_job(ds, df, BASE_MODELS, "holdout", 0, [], src, False)
            if JOB["status"] == "error":
                return
            body = JOB["body"].split('<p><a href="/">')[0]
            parts.append(body.replace("<h1>Ket qua thuc nghiem</h1>", f"<h1>{title}</h1>", 1))
        if ucol:
            JOB["current"] = "TF-IDF ky tu (kiem tra URL)"
            add_tfidf_model(ds, df)
        nav = '<p><a href="/">&larr; Trang chu</a>' + (' | <a href="/check_url">Kiem tra URL</a>' if ucol else "") + \
              ' | <a href="/history">Lich su</a></p>'
        JOB["body"] = "".join(parts) + nav
        JOB["status"] = "done"
    except Exception as e:
        JOB.update(status="error", error=str(e))

def start_auto(ds):
    if JOB["status"] == "running":
        return False
    JOB.update(status="running", done=0, total=1, current="Chuan bi...", error="", body="")
    threading.Thread(target=auto_job, args=(ds,), daemon=True).start()
    return True

@app.route("/run", methods=["POST"])
def run():
    ds = CURRENT["name"]
    if ds not in DATASETS:
        return err_page("Chua co dataset.")
    if META.get(ds, {}).get("confirmed") is False:
        return err_page(NEED_CONFIRM)
    start_auto(ds)
    return redirect("/progress")

@app.route("/progress")
def progress():
    if JOB["status"] == "done":
        return redirect("/run/result")
    if JOB["status"] == "error":
        return err_page(JOB["error"])
    if JOB["status"] == "idle":
        return redirect("/")
    pct = int(100 * JOB["done"] / max(JOB["total"], 1))
    return page(f"""<div class="card"><h1>Dang chay thuc nghiem...</h1>
    <p>Dang huan luyen: <b>{esc(JOB['current'] or '...')}</b> ({JOB['done']}/{JOB['total']} mo hinh xong)</p>
    <div class="bar"><div style="width:{pct}%"></div></div>
    <p class="hint">Trang tu lam moi moi 2 giay. Co the cho nhieu phut voi dataset lon.</p></div>""", refresh=True)

@app.route("/run/result")
def run_result():
    return page(JOB["body"]) if JOB["body"] else redirect("/")


# ---------------- Lich su ----------------
@app.route("/history")
def history_page():
    if not HISTORY:
        return page("<div class='card'><h1>Chua co ket qua nao</h1><a href='/'>&larr; Trang chu</a></div>")
    h = pd.DataFrame(HISTORY)
    sel = request.args.get("dataset", "")
    if sel:
        h = h[h["Dataset"] == sel]
    d = h.copy()
    for c in ["Accuracy", "Precision", "Recall", "F1", "F1-score", "ROC_AUC"]:
        if c in d.columns:
            d[c] = (d[c] * 100).round(2).astype(str) + "%"
    if "Train_s" in d.columns:
        d["Train_s"] = d["Train_s"].round(2)
    d = d.sort_values("Thoi_gian", ascending=False)
    opts = "<option value=''>-- Tat ca dataset --</option>" + "".join(
        f"<option value='{esc(x)}' {'selected' if x == sel else ''}>{esc(x)}</option>" for x in pd.DataFrame(HISTORY)["Dataset"].unique())
    return page(f"""<h1>Lich su thuc nghiem</h1><div class="card">
    <form method="get"><label>Loc theo dataset: <select name="dataset" onchange="this.form.submit()">{opts}</select></label></form>
    <p class="hint">{len(h)} ket qua.</p>{d.to_html(index=False, escape=True, na_rep='-')}</div>
    <div class="card"><a href="/history/csv">Tai history.csv</a><br><br>
    <form method="post" action="/history/clear" onsubmit="return confirm('Xoa toan bo lich su?');">
    <button class="danger">Xoa toan bo lich su</button></form></div><p><a href="/">&larr; Trang chu</a></p>""")

@app.route("/history/csv")
def history_csv():
    from flask import Response
    return Response(pd.DataFrame(HISTORY).to_csv(index=False), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=history.csv"})

@app.route("/history/clear", methods=["POST"])
def clear_history():
    HISTORY.clear()
    save_history()
    return redirect("/history")


# ---------------- Kiem tra URL ----------------
def bundle_path(ds):
    return os.path.join(MODEL_DIR, secure_filename(ds) + ".joblib")

def get_bundle(ds):
    if ds not in URL_BUNDLES and os.path.exists(bundle_path(ds)):
        URL_BUNDLES[ds] = joblib.load(bundle_path(ds))
    b = URL_BUNDLES.get(ds)
    return b if b and b.get("version") == FEATURE_VERSION else None

@app.route("/check_url", methods=["GET", "POST"])
def check_url():
    ds = CURRENT["name"]
    b = get_bundle(ds) if ds else None
    if not b:
        return err_page("Dataset hien tai chua co mo hinh kiem tra URL. Hay bam Chay lai thuc nghiem o trang chu.")
    result = ""
    if request.method == "POST":
        url = request.form.get("url", "").strip()
        feat = pd.DataFrame([extract_basic_features(url)])
        rows, ph, preds = "", 0, {}
        for name, (kind, m) in b["models"].items():
            X = feat if kind == "feat" else pd.Series([normalize_url(url)])
            pred = int(m.predict(X)[0])
            ph += pred
            preds[name] = pred
            pr = f" ({m.predict_proba(X)[0][pred]*100:.1f}%)" if hasattr(m, "predict_proba") else ""
            color = "#f87171" if pred else "#4ade80"
            rows += (f"<tr><td>{esc(name)}</td><td style='color:{color};font-weight:bold'>{'PHISHING' if pred else 'Hop le'}{pr}</td>"
                     f"<td>{b['accs'][name]*100:.2f}%</td></tr>")
        n = len(b["models"])
        bad = ph > n / 2
        result = f"""<div class="box"><div class="big" style="color:{'#f87171' if bad else '#4ade80'}">{'NGHI NGO PHISHING' if bad else 'CO VE AN TOAN'}</div>
        <div>{ph}/{n} mo hinh cho la phishing</div>
        <div class='hint'>URL sau chuan hoa: {esc(normalize_url(url))}</div>
        {f"<div class='hint'>Mo hinh tot nhat theo F1 ({esc(b['best'])}): {'PHISHING' if preds.get(b['best']) else 'Hop le'}</div>" if b.get("best") in preds else ""}</div>
        <table><tr><th>Mo hinh</th><th>Du doan</th><th>Accuracy (tap test)</th></tr>{rows}</table>
        <h2>Dac trung trich tu URL</h2><table><tr><th>Dac trung</th><th>Gia tri</th></tr>
        {''.join(f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in feat.iloc[0].items())}</table>"""
    return page(f"""<h1>Kiem tra URL cu the</h1><p class="hint">Mo hinh huan luyen tu dataset: <b>{esc(ds)}</b></p>
    <div class="card"><form method="post"><input type="text" name="url" style="width:85%" required
    placeholder="Dan URL can kiem tra"> <button>Kiem tra</button></form>{result}
    <p><a href="/">&larr; Trang chu</a></p></div>""")


try:
    if MASTER_NAME not in DATASETS:
        frames = []
        for _name, _df in list(DATASETS.items()):
            if _name != MASTER_NAME and "label" in _df.columns:
                _uc = find_url_column(_df)
                if _uc:
                    _z = pd.DataFrame({"url": _df[_uc].astype(str), "label": pd.to_numeric(_df["label"], errors="coerce")}).dropna(subset=["label"])
                    _z["label"] = _z["label"].astype(int); frames.append(_z)
        if frames:
            _m = pd.concat(frames, ignore_index=True); _m["url"] = _m["url"].map(normalize_url)
            _m = _m[_m["url"].str.len() > 0].drop_duplicates("url", keep="last").reset_index(drop=True)
            _m.to_csv(MASTER_FILE, index=False); DATASETS[MASTER_NAME] = _m
            META[MASTER_NAME] = {"phishing_value":"1", "confirmed":True, "accumulative":True}; save_meta()
            CURRENT["name"] = MASTER_NAME
except Exception:
    pass

if __name__ == "__main__":
    app.run(debug=False, threaded=True, port=5001)