import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

# 1. Đường dẫn & Tham số
current_dir = os.path.dirname(os.path.abspath(__file__))
TEST_PATH = os.path.join(current_dir, 'dataset', 'test')
MODEL_PATH = os.path.join(current_dir, 'trained_model', 'efficientnet_b0_fruit_veg.h5')

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# 2. Load tập Test (Lưu ý: shuffle=False để không làm lệch thứ tự nhãn)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

class_labels = list(test_generator.class_indices.keys())

# 3. Load mô hình đã lưu
print(f"\n--- Đang nạp mô hình từ: {MODEL_PATH} ---")
model = load_model(MODEL_PATH)

# 4. Tiến hành Dự đoán trên Tập Test
print("\n--- Đang thực hiện dự đoán trên tập Test... ---")
y_pred_probs = model.predict(test_generator, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1) # Chọn lớp có xác suất cao nhất
y_true = test_generator.classes          # Nhãn thực tế

# 5. Tính toán các chỉ số Đánh giá Tổng quan (Overall Metrics)
acc = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

print("\n" + "="*50)
print("     KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH EFFICIENTNETB0")
print("="*50)
print(f"📌 Test Accuracy : {acc * 100:.2f}%")
print(f"📌 Macro Precision: {precision * 100:.2f}%")
print(f"📌 Macro Recall   : {recall * 100:.2f}%")
print(f"📌 Macro F1-Score : {f1 * 100:.2f}%")
print("="*50)

# 6. In Báo cáo Phân loại Chi tiết cho 36 lớp
print("\n--- Báo cáo Chi tiết từng Lớp (Classification Report) ---")
print(classification_report(y_true, y_pred, target_names=class_labels))

# 7. Vẽ Confusion Matrix (Ma trận Nhầm lẫn)
print("\n--- Đang vẽ Confusion Matrix... ---")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(18, 14))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.title('Confusion Matrix - EfficientNetB0', fontsize=16)
plt.xlabel('Nhãn Dự đoán (Predicted)', fontsize=12)
plt.ylabel('Nhãn Thực tế (True)', fontsize=12)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

# Lưu biểu đồ Confusion Matrix để đưa vào báo cáo Word/Slide
cm_save_path = os.path.join(current_dir, 'confusion_matrix_efficientnet.png')
plt.savefig(cm_save_path, dpi=300)
print(f"Đã lưu hình Confusion Matrix tại: {cm_save_path}")
plt.show()
