import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# ---------------------------------------------------------
# 1. CẤU HÌNH TRANG STREAMLIT
# ---------------------------------------------------------
st.set_page_config(
    page_title="Nhận diện Rau Củ Quả",
    page_icon="🥦",
    layout="centered"
)

# Danh sách 36 nhãn rau củ quả (Theo đúng thứ tự alphabet trong thư mục train)
CLASS_NAMES = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum',
    'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant',
    'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce',
    'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple',
    'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn',
    'sweetpotato', 'tomato', 'turnip', 'watermelon'
]

# ---------------------------------------------------------
# 2. NẠP MÔ HÌNH DÃ LƯU (SỬ DỤNG CACHE ĐỂ TỐI ƯU TỐC ĐỘ)
# ---------------------------------------------------------
@st.cache_resource
def load_efficientnet_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'trained_model', 'efficientnet_b0_fruit_veg.h5')
    model = load_model(model_path)
    return model

# Load model khi khởi chạy app
with st.spinner("Đang tải mô hình AI... Vui lòng chờ trong giây lát!"):
    model = load_efficientnet_model()

# ---------------------------------------------------------
# 3. GIAO DIỆN NGƯỜI DÙNG (UI)
# ---------------------------------------------------------
st.title("🥦 Phân Loại Rau Củ Quả")
st.write("Mô hình sử dụng: **EfficientNetB0**")
st.markdown("---")

# Bộ chọn file ảnh từ máy tính
uploaded_file = st.file_uploader(
    "Chọn một bức ảnh rau củ quả từ máy tính của bạn...",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    # Hiển thị ảnh đã chọn
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Ảnh bạn đã tải lên", use_container_width=True)

    # Nút bấm dự đoán
    if st.button("🔍 Dự đoán loại Rau Củ Quả", type="primary"):
        with st.spinner("Đang phân tích hình ảnh..."):
            # A. TIỀN XỬ LÝ ÁNH ĐẦU VÀO
            # 1. Resize ảnh về (224, 224)
            img_resized = image.resize((224, 224))

            # 2. Chuyển thành mảng Numpy
            img_array = np.array(img_resized)

            # 3. Thêm chiều batch (1, 224, 224, 3)
            img_batch = np.expand_dims(img_array, axis=0)

            # 4. Normalize chuẩn EfficientNetB0
            img_preprocessed = preprocess_input(img_batch)

            # B. DỰ ĐOÁN
            predictions = model.predict(img_preprocessed)
            score = tf.nn.softmax(predictions[0]) # Hoặc dùng trực tiếp predictions[0] nếu output đã là softmax

            predicted_class_idx = np.argmax(predictions[0])
            predicted_class_name = CLASS_NAMES[predicted_class_idx]
            confidence = predictions[0][predicted_class_idx] * 100

            # C. HIỂN THỊ KẾT QUẢ
            st.success(f"**Kết quả dự đoán:** {predicted_class_name.upper()}")
            st.info(f"**Độ tin cậy (Confidence):** {confidence:.2f}%")

            # Hiển thị Top-3 dự đoán cao nhất
            st.write("---")
            st.write("📊 **Top 3 khả năng cao nhất:**")
            top_3_indices = np.argsort(predictions[0])[-3:][::-1]

            for idx in top_3_indices:
                class_name = CLASS_NAMES[idx]
                prob = predictions[0][idx] * 100
                st.write(f"- **{class_name.capitalize()}**: {prob:.2f}%")
                st.progress(int(prob))
