import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# 1. Đường dẫn thư mục
current_dir = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(current_dir, 'dataset', 'train')
VAL_PATH   = os.path.join(current_dir, 'dataset', 'validation')
TEST_PATH  = os.path.join(current_dir, 'dataset', 'test')

# 2. Đếm số lượng ảnh trong từng tập dữ liệu
def count_files(dir_path):
    categories = os.listdir(dir_path)
    data = {}
    for cat in categories:
        cat_path = os.path.join(dir_path, cat)
        if os.path.isdir(cat_path):
            data[cat] = len(os.listdir(cat_path))
    return data

train_counts = count_files(TRAIN_PATH)
val_counts = count_files(VAL_PATH)
test_counts = count_files(TEST_PATH)

df_stats = pd.DataFrame({
    'Train': train_counts,
    'Validation': val_counts,
    'Test': test_counts
}).sort_values(by='Train', ascending=False)

print(f"Tổng số lớp (classes): {len(df_stats)}")
print(f"Tổng số ảnh Train: {df_stats['Train'].sum()}")
print(f"Tổng số ảnh Val: {df_stats['Validation'].sum()}")
print(f"Tổng số ảnh Test: {df_stats['Test'].sum()}")
print("\nBảng thống kê 5 lớp đầu tiên:")
print(df_stats.head())

# 3. Trực quan hóa Phân bố Dữ liệu (Class Distribution in Train set)
plt.figure(figsize=(15, 6))
sns.barplot(x=df_stats.index, y=df_stats['Train'], palette='viridis')
plt.xticks(rotation=90)
plt.title('Phân bố số lượng ảnh trong tập Train theo từng loại Rau Củ Quả')
plt.xlabel('Tên loại Rau Củ Quả')
plt.ylabel('Số lượng ảnh')
plt.tight_layout()
plt.show()

# 4. Hiển thị mẫu ảnh ngẫu nhiên
def plot_sample_images(base_path, rows=3, cols=5):
    fig, axes = plt.subplots(rows, cols, figsize=(15, 9))
    categories = os.listdir(base_path)

    # Chọn ngẫu nhiên các lớp
    selected_cats = np.random.choice(categories, size=rows*cols, replace=False)

    for idx, cat in enumerate(selected_cats):
        cat_dir = os.path.join(base_path, cat)
        img_name = np.random.choice(os.listdir(cat_dir))
        img_path = os.path.join(cat_dir, img_name)

        img = Image.open(img_path)

        ax = axes[idx // cols, idx % cols]
        ax.imshow(img)
        ax.set_title(f"{cat}\n{img.size[0]}x{img.size[1]}px")
        ax.axis('off')

    plt.suptitle('Mẫu ảnh ngẫu nhiên trong Dataset (Kèm kích thước gốc)', fontsize=16)
    plt.tight_layout()
    plt.show()

plot_sample_images(TRAIN_PATH)
