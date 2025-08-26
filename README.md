# Dự án MLflow: Phân loại với dữ liệu `make_classification` và Flask

Mục tiêu
- Sinh dữ liệu bằng `sklearn.datasets.make_classification` cho bài toán phân loại nhị phân.
- Huấn luyện mô hình cơ bản. Thử nhiều siêu tham số và nhiều mô hình.
- Ghi log vào MLflow. So sánh kết quả giữa các lần thử.
- Tìm mô hình tốt nhất theo F1 hoặc Accuracy. Đăng ký vào Model Registry.
- Tạo ứng dụng Flask dùng mô hình tốt nhất ở stage Production.

## 1. Cài đặt môi trường cục bộ

```bash
python -m venv .venv
source .venv/bin/activate            # Windows dùng: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Khởi chạy MLflow Server với registry cục bộ

Mở một cửa sổ terminal khác, chạy:

```bash
./start_mlflow_server.sh
```

Server mặc định chạy ở `http://127.0.0.1:5000`. Cửa sổ này cần mở trong lúc chạy thí nghiệm.

## 3. Chạy tuning và đăng ký mô hình tốt nhất

```bash
# ví dụ chạy với lưới tham số mặc định
python src/tune.py --experiment-name "cls-exp" --metric f1

# hoặc tùy biến số mẫu và độ tách lớp
python src/tune.py --experiment-name "cls-exp" --metric f1 --n-samples 3000 --class-sep 1.5
```

Kịch bản sẽ:
- Sinh dữ liệu, chia train test.
- Thử LogisticRegression và RandomForest với nhiều siêu tham số.
- Log tất cả thông tin vào MLflow.
- Chọn run tốt nhất theo metric chỉ định. Đăng ký thành model `BestClassifier`.
- Chuyển version tốt nhất sang stage Production. Các version cũ được chuyển sang Archived.

Xem giao diện MLflow tại `http://127.0.0.1:5000`.

## 4. So sánh và xuất bảng kết quả

```bash
python src/compare.py --experiment-name "cls-exp" --metric f1
```

Kết quả hiển thị dạng bảng. Đồng thời lưu ra `results.csv` ở thư mục gốc.

## 5. Chạy Flask app dùng mô hình tốt nhất

```bash
export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
python src/app.py
```

Truy cập `http://127.0.0.1:8000`.

- POST `/predict` với JSON:
```json
{
  "X": [[0.2, 1.3, -0.7, ...], [1.1, -0.3, 0.1, ...]]
}
```
hoặc
```json
{
  "features": {"f0": 0.2, "f1": 1.3, "f2": -0.7, "...": 0.0}
}
```

- GET `/` đưa ra hướng dẫn nhanh.

## 6. File quan trọng

- `src/tune.py` thực hiện sinh dữ liệu, thử tham số, chọn mô hình tốt nhất, đăng ký vào registry.
- `src/train_one.py` huấn luyện một lần và log vào MLflow.
- `src/compare.py` trích xuất bảng so sánh các run.
- `src/app.py` Flask sử dụng model ở `models:/BestClassifier/Production`.
- `start_mlflow_server.sh` chạy MLflow server với SQLite registry.
- `requirements.txt` các thư viện cần thiết.

Gợi ý chấm điểm
- 1 điểm cho dữ liệu và pipeline cơ bản.
- 2 điểm cho ghi log MLflow đầy đủ params, metrics, artifacts.
- 2 điểm cho tuning nhiều mô hình và so sánh kết quả.
- 2 điểm cho đăng ký và chuyển stage mô hình tốt nhất.
- 1 điểm cho Flask app hoạt động đúng với mô hình Production.