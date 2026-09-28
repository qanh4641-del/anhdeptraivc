import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# --- CẤU HÌNH LIÊN KẾT GOOGLE DRIVE ---
# Thay đổi chuỗi bên dưới thành ID file của bạn trên Google Drive.
# Ví dụ link: https://drive.google.com/file/d/1abc123XYZ.../view?usp=sharing
# Thì ID chính là đoạn: 1abc123XYZ...
MODEL_DRIVE_ID = "1nAlUWEsaxqOSGEP23nJxn9h3VQYU441L"
MODEL_FILENAME = "EfficientNetB7_CatDog.h5"

# Cấu hình trang
st.set_page_config(page_title="AI Nhận diện Chó Mèo", page_icon="🐶", layout="centered")

st.title("🐶🐱 Trợ lý AI Nhận diện Chó & Mèo")
st.markdown("Tải một bức ảnh lên và AI (EfficientNetB7) sẽ cho bạn biết đó là chó hay mèo!")

# Hàm tải file từ Drive
def download_model_from_drive(file_id, output_name):
    # Chỉ tải xuống nếu file chưa tồn tại trên máy chủ (tiết kiệm thời gian)
    if not os.path.exists(output_name):
        url = f'https://drive.google.com/uc?id={file_id}'
        try:
            # gdown giúp tải file dung lượng lớn bỏ qua cảnh báo diệt virus của Google
            gdown.download(url, output_name, quiet=False)
        except Exception as e:
            st.error(f"Lỗi khi tải mô hình từ Drive: {e}")
            return None
    return output_name

# Hàm tải mô hình vào bộ nhớ AI (Dùng cache_resource để load model 1 lần duy nhất)
@st.cache_resource
def load_ai_model():
    # 1. Gọi hàm tải file từ Drive về máy chủ trước
    model_path = download_model_from_drive(MODEL_DRIVE_ID, MODEL_FILENAME)
    
    # 2. Đọc file model bằng TensorFlow
    if model_path and os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        return model
    return None

# Khởi tạo model
try:
    with st.spinner("Đang tải và chuẩn bị mô hình AI (Lần đầu có thể mất 1-3 phút)..."):
        model = load_ai_model()
        if model is None:
            st.error("⚠️ KHÔNG TÌM THẤY MÔ HÌNH AI!")
            st.info("Hãy kiểm tra lại MODEL_DRIVE_ID và quyền chia sẻ file trên Google Drive.")
            st.stop()
except Exception as e:
    st.error(f"⚠️ CÓ LỖI XẢY RA TRONG QUÁ TRÌNH TẢI MÔ HÌNH: {e}")
    st.stop() # Dừng chạy app nếu không có model

# Tạo giao diện upload ảnh
uploaded_file = st.file_uploader("Chọn một bức ảnh (Định dạng: JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mở và hiển thị ảnh
    image = Image.open(uploaded_file)
    # Hiển thị ảnh ở giữa trang
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Ảnh bạn đã tải lên", use_container_width=True)
    
    # Nút bấm dự đoán
    if st.button("🔍 Phân tích ảnh ngay", use_container_width=True):
        with st.spinner("AI đang suy nghĩ..."):
            try:
                # --- TIỀN XỬ LÝ ẢNH ---
                # 1. Chuyển ảnh về kích thước 224x224 giống lúc huấn luyện
                img_resized = image.resize((224, 224))
                
                # 2. Chuyển thành mảng numpy và lấy đủ 3 kênh màu (RGB)
                if img_resized.mode != 'RGB':
                    img_resized = img_resized.convert('RGB')
                
                img_array = np.array(img_resized)
                
                # 3. Chuẩn hóa pixel (rescale 1./255)
                img_array = img_array / 255.0
                
                # 4. Thêm chiều batch (batch_size, height, width, channels)
                img_array = np.expand_dims(img_array, axis=0)
                
                # --- DỰ ĐOÁN ---
                prediction = model.predict(img_array)[0][0]
                
                # Trong ImageDataGenerator: thư mục 'cats' đứng trước 'dogs' (theo bảng chữ cái)
                # Lớp 0 = Mèo (Gần 0.0), Lớp 1 = Chó (Gần 1.0)
                st.markdown("---")
                if prediction < 0.5:
                    confidence = (1 - prediction) * 100
                    st.success(f"🎉 **Kết quả:** AI dự đoán đây là **MÈO 🐱**")
                    st.info(f"Độ tin cậy: **{confidence:.2f}%**")
                else:
                    confidence = prediction * 100
                    st.success(f"🎉 **Kết quả:** AI dự đoán đây là **CHÓ 🐶**")
                    st.info(f"Độ tin cậy: **{confidence:.2f}%**")
                    
                st.progress(int(confidence))
                    
            except Exception as e:
                st.error(f"Đã xảy ra lỗi khi phân tích ảnh: {e}")
