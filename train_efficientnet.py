import os
import matplotlib.pyplot as plt
# Ẩn các thông báo log thừa của TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ==========================================
# 1. THIẾT LẬP ĐƯỜNG DẪN & SIÊU THAM SỐ
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(current_dir, 'dataset', 'train')
VAL_PATH   = os.path.join(current_dir, 'dataset', 'validation')
TEST_PATH  = os.path.join(current_dir, 'dataset', 'test')

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 36

# ==========================================
# 2. NAP DỮ LIỆU VỚI DATA AUGMENTATION
# ==========================================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

print("--- Nạp tập Train ---")
train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True
)

print("--- Nạp tập Validation ---")
val_generator = val_datagen.flow_from_directory(
    VAL_PATH, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)

print("--- Nạp tập Test ---")
test_generator = test_datagen.flow_from_directory(
    TEST_PATH, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)

# ==========================================
# 3. XÂY DỰNG MÔ HÌNH EFFICIENTNETB0 (TRANSFER LEARNING)
# ==========================================
print("\n--- Khởi tạo Mô hình EfficientNetB0 Base ---")
# Khởi tạo Base Model với trọng số pretrained ImageNet
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,                  # Bỏ lớp phân loại 1000 lớp gốc của ImageNet
    input_shape=(224, 224, 3)
)

# Khóa (Freeze) toàn bộ các lớp của Base Model trong giai đoạn 1
base_model.trainable = False

# Xây dựng mô hình phân loại mới (Classification Head)
inputs = layers.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x) # Thu gom đặc trưng
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)             # Chống Overfitting
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x) # Output 36 lớp

model = models.Model(inputs, outputs)

model.summary()

# ==========================================
# 4. GIAI ĐOẠN 1: TRAIN FEATURE EXTRACTION
# ==========================================
print("\n========== GIAI ĐOẠN 1: TRAIN FEATURE EXTRACTION ==========")
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Đặt các hàm Callbacks để tự động quản lý quá trình Train
callbacks_phase1 = [
    ModelCheckpoint(os.path.join(current_dir, 'best_efficientnet_phase1.h5'), monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
]

EPOCHS_PHASE1 = 10
history_phase1 = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE1,
    validation_data=val_generator,
    callbacks=callbacks_phase1
)

# ==========================================
# 5. GIAI ĐOẠN 2: FINE-TUNING
# ==========================================
print("\n========== GIAI ĐOẠN 2: FINE-TUNING MODEL ==========")
# Mở xả (Unfreeze) toàn bộ Base Model
base_model.trainable = True

# Chỉ mở xả 30 lớp cuối cùng của EfficientNetB0 để fine-tune, đóng các lớp đầu
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Compile lại với Learning Rate RẤT NHỎ (1e-5) để không làm hỏng weights đã học
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks cho Fine-tuning
callbacks_phase2 = [
    ModelCheckpoint(os.path.join(current_dir, 'efficientnet_b0_fruit_veg.h5'), monitor='val_accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1),
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
]

EPOCHS_PHASE2 = 15
history_phase2 = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE2,
    validation_data=val_generator,
    callbacks=callbacks_phase2
)

print("\nĐã huấn luyện xong! Model tốt nhất được lưu tại: efficientnet_b0_fruit_veg.h5")

# ==========================================
# 6. VẼ & LƯU BIỂU ĐỒ ACCURACY / LOSS
# ==========================================
# Nối lịch sử huấn luyện của Giai đoạn 1 và Giai đoạn 2
acc = history_phase1.history['accuracy'] + history_phase2.history['accuracy']
val_acc = history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy']

loss = history_phase1.history['loss'] + history_phase2.history['loss']
val_loss = history_phase1.history['val_loss'] + history_phase2.history['val_loss']

epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(14, 5))

# Biểu đồ Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy', color='blue', linewidth=2)
plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='orange', linewidth=2)
plt.axvline(x=len(history_phase1.history['accuracy']), color='green', linestyle='--', label='Start Fine-Tuning')
plt.title('EfficientNetB0 - Training & Validation Accuracy', fontsize=12)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)

# Biểu đồ Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss', color='blue', linewidth=2)
plt.plot(epochs_range, val_loss, label='Validation Loss', color='orange', linewidth=2)
plt.axvline(x=len(history_phase1.history['accuracy']), color='green', linestyle='--', label='Start Fine-Tuning')
plt.title('EfficientNetB0 - Training & Validation Loss', fontsize=12)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()

# Lưu biểu đồ về đĩa cứng để chèn vào Báo cáo & Slide
chart_path = os.path.join(current_dir, 'accuracy_loss_efficientnet.png')
plt.savefig(chart_path, dpi=300)
print(f"\n[OK] Đã vẽ và lưu biểu đồ Accuracy/Loss tại: {chart_path}")
plt.show()
