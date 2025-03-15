# Mô hình ANN phân loại tình trạng thất nghiệp

## 1. Giới thiệu
Dự án này sử dụng mạng nơ-ron nhân tạo (ANN) để phân loại tình trạng thất nghiệp của nhân viên dựa trên các yếu tố như trình độ học vấn, kinh nghiệm, ngành nghề, mức thu nhập, và nhiều đặc điểm khác.

## 2. Dữ liệu
Tập dữ liệu sử dụng là **WA_Fn-UseC_-HR-Employee-Attrition.csv**, được lấy từ trang web [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset/data), bao gồm các thông tin sau:
- **Biến đầu ra (Label)**: `Attrition` (Tình trạng nghỉ việc: 1 - Có, 0 - Không)
- **Biến đầu vào (Features)**: `Age`, `BusinessTravel`, `Department`, `Education`, `JobRole`, `MonthlyIncome`, ...

## 3. Các bước thực hiện
### 3.1. Khám phá và tiền xử lý dữ liệu
- **Trực quan hóa dữ liệu**: Biểu đồ phân phối `Attrition`, độ tuổi theo `Attrition`, hộp số thu nhập hàng tháng.
- **Làm sạch dữ liệu**:
  - Xử lý giá trị thiếu
  - Loại bỏ dữ liệu trùng lặp
  - Kiểm tra kiểu dữ liệu
  - Loại bỏ outliers bằng IQR
  - Chuẩn hóa dữ liệu số
  - Mã hóa dữ liệu phân loại (Label Encoding)

### 3.2. Xây dựng mô hình ANN
Ba mô hình được triển khai, có thêm **Dropout** để giảm overfitting và **Early Stopping** để dừng huấn luyện khi mô hình không còn cải thiện:
- **Model 1**: 2 lớp ẩn (64-32 neuron, ReLU, Dropout 0.2)
- **Model 2**: 3 lớp ẩn (128-64-32 neuron, ReLU, Dropout 0.3, sâu hơn)
- **Model 3**: 2 lớp ẩn (64-64-32-16 neuron, Tanh, Dropout 0.2)

Tất cả mô hình sử dụng:
- `binary_crossentropy` làm hàm mất mát
- `Adam` làm optimizer
- `EarlyStopping` với `patience=5` để ngăn overfitting

### 3.3. Đánh giá mô hình
- So sánh **độ chính xác** trên tập kiểm tra.
- So sánh **hàm mất mát** để kiểm tra khả năng tổng quát hóa của mô hình.

## 4. Kết quả và Nhận xét
- **Model 2 (128-64-32 ReLU, Dropout 0.3, EarlyStopping)** có độ chính xác cao nhất.
- **Model 3 (Tanh, Dropout 0.2)** không hoạt động tốt bằng ReLU.
- Việc sử dụng Dropout và Early Stopping giúp giảm overfitting, mô hình tổng quát hóa tốt hơn.

## 5. Hướng phát triển
- Thử nghiệm thêm các kiến trúc ANN khác.
- Áp dụng các phương pháp giảm overfitting (L1/L2 Regularization, Batch Normalization).
- So sánh với các mô hình máy học khác như Random Forest, XGBoost.

## 6. Cách chạy chương trình
1. Cài đặt các thư viện cần thiết:
   ```bash
   pip install tensorflow pandas numpy seaborn matplotlib scikit-learn
   ```
2. Chạy script Python:
   ```bash
   python ann_unemployment.py
   ```
3. Quan sát kết quả đánh giá mô hình.

## 7. Công nghệ sử dụng
- **Python** với thư viện:
  - `TensorFlow`, `Keras`: Xây dựng mô hình ANN
  - `Pandas`, `NumPy`: Xử lý dữ liệu
  - `Seaborn`, `Matplotlib`: Trực quan hóa dữ liệu
  - `Scikit-learn`: Chia dữ liệu, chuẩn hóa, mã hóa dữ liệu
