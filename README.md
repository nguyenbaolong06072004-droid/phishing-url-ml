Phishing URL Machine Learning Demo
Ứng dụng web local dùng Flask để thực nghiệm và so sánh các mô hình Machine Learning trên dữ liệu phishing URL.
1.Cài thư viện
python 3.11
Trong CMD, tại thư mục project:
chạy:
python -m venv venv
venv311\Scripts\activate
Chạy:
pip install flask pandas numpy scikit-learn xgboost matplotlib seaborn ucimlrepo

2. Chạy chương trình
Trong CMD, tại thư mục project:
chạy: 
python app.py

Truy cập
http://127.0.0.1:5001

3. Sử dụng chương trình

Bước 1 — Nạp dữ liệu

Có thể:

Sử dụng dataset UCI Phishing Websites có sẵn trong chương trình.

Upload datas

Nếu upload dataset riêng, file cần có một cột nhãn.

Mặc định tên cột nhãn là:

label

Nếu tên cột khác, nhập tên cột đó vào ô Tên cột nhãn.

Nhãn được hỗ trợ:

0 / 1

hoặc:

-1 / 1

Trong chương trình, -1 sẽ được chuyển thành 0.

Quy ước:

0 = Hợp lệ
1 = Phishing

Bước 2 — Chọn mô hình

Có thể chọn:

Logistic Regression

Decision Tree

SVM (RBF)

Random Forest

XGBoost

Bước 3 — Chạy thực nghiệm

Nhấn:

Chạy thực nghiệm

Chương trình sẽ huấn luyện và đánh giá các mô hình đã chọn.

Bước 4 — Xem kết quả

Kết quả gồm:

Accuracy

Precision

Recall

F1-score

Biểu đồ so sánh mô hình

Confusion Matrix

Top 10 đặc trưng quan trọng nhất của Random Forest nếu chọn Random Forest

