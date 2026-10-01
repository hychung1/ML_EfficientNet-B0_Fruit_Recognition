import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

# 1. Đường dẫn tương đối đồng bộ với file eda.py
current_dir = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(current_dir, 'dataset', 'train')
VAL_PATH   = os.path.join(current_dir, 'dataset', 'validation')
TEST_PATH  = os.path.join(current_dir, 'dataset', 'test')

# 2. Cấu hình Tham số chung
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# 3. Tạo Data Generator cho từng tập dữ liệu

# Tập Train: Áp dụng Data Augmentation + EfficientNet Preprocessing
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, # Normalize chuẩn cho EfficientNetB0
    rotation_range=30,                        # Xoay ảnh ngẫu nhiên trong khoảng [-30, 30] độ
    width_shift_range=0.2,                    # Dịch chuyển chiều ngang 20%
    height_shift_range=0.2,                   # Dịch chuyển chiều dọc 20%
    shear_range=0.2,                          # Cắt nghiêng ảnh
    zoom_range=0.2,                           # Phóng to / thu nhỏ ảnh 20%
    horizontal_flip=True,                     # Lật ngang ảnh ngẫu nhiên
    fill_mode='nearest'                       # Điền khoảng trống sinh ra khi biến đổi ảnh
)

# Tập Validation & Test: CHỈ làm preprocess_input (Không Augment)
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# 4. Load dữ liệu từ thư mục bằng flow_from_directory
print("--- Nạp dữ liệu tập Train ---")
train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMAGE_SIZE,       # Resize về 224x224
    batch_size=BATCH_SIZE,
    class_mode='categorical',     # Phân loại đa lớp (One-hot encoding)
    shuffle=True
)

print("\n--- Nạp dữ liệu tập Validation ---")
val_generator = val_datagen.flow_from_directory(
    VAL_PATH,
    target_size=IMAGE_SIZE,       # Resize về 224x224
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False                 # Không shuffle ở tập Val để dễ theo dõi
)

print("\n--- Nạp dữ liệu tập Test ---")
test_generator = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=IMAGE_SIZE,       # Resize về 224x224
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False                 # Đặt shuffle=False để phục vụ đánh giá (Confusion Matrix)
)

# 5. Kiểm tra danh sách nhãn lớp thu được
class_indices = train_generator.class_indices
labels = list(class_indices.keys())
print(f"\nĐã nạp thành công {len(labels)} lớp rau củ quả.")
