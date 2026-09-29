import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# --- CẤU HÌNH LIÊN KẾT GOOGLE DRIVE ---
MODEL_DRIVE_ID = "1nAlUWEsaxqOSGEP23nJxn9h3VQYU441L"
MODEL_FILENAME = "EfficientNetB7_CatDog.h5"

# Cấu hình trang
st.set_page_config(page_title="AI Nhận diện Chó Mèo", page_icon="🐶", layout="centered")

st.title("🐶🐱 Trợ lý AI Nhận diện Chó & Mèo")
st.markdown("Tải một bức ảnh lên và AI (EfficientNetB7) sẽ cho bạn biết đó là chó, mèo hay loài khác!")

# Hàm tải file từ Drive
def download_model_from_drive(file_id, output_name):
    if not os.path.exists(output_name):
        url = f'https://drive.google.com/uc?id={file_id}'
        try:
            gdown.download(url, output_name, quiet=False)
        except Exception as e:
            st.error(f"Lỗi khi tải mô hình từ Drive: {e}")
            return None
    return output_name

# Hàm tải mô hình vào bộ nhớ AI
@st.cache_resource
def load_ai_model():
    model_path = download_model_from_drive(MODEL_DRIVE_ID, MODEL_FILENAME)
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
    st.stop() 

# Tạo giao diện upload ảnh
uploaded_file = st.file_uploader("Chọn một bức ảnh (Định dạng: JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Ảnh bạn đã tải lên", use_container_width=True)
    
    if st.button("🔍 Phân tích ảnh ngay", use_container_width=True):
        with st.spinner("AI đang suy nghĩ..."):
            try:
                # --- TIỀN XỬ LÝ ẢNH ---
                img_resized = image.resize((224, 224))
                
                if img_resized.mode != 'RGB':
                    img_resized = img_resized.convert('RGB')
                
                img_array = np.array(img_resized)
                img_array = img_array / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # --- DỰ ĐOÁN ---
                prediction = model.predict(img_array)[0][0]
                
                # Tính phần trăm độ tin cậy cho từng lớp
                confidence_cat = (1 - prediction) * 100
                confidence_dog = prediction * 100
                
                st.markdown("---")
                
                # NGƯỠNG TIN CẬY (Threshold): Có thể điều chỉnh số 75.0 cao hay thấp tùy ý
                THRESHOLD = 50.0 
                
                if confidence_cat >= THRESHOLD:
                    st.success(f"🎉 **Kết quả:** AI dự đoán đây là **MÈO 🐱**")
                    st.info(f"Độ tin cậy: **{confidence_cat:.2f}%**")
                    st.progress(int(confidence_cat))
                
                elif confidence_dog >= THRESHOLD:
                    st.success(f"🎉 **Kết quả:** AI dự đoán đây là **CHÓ 🐶**")
                    st.info(f"Độ tin cậy: **{confidence_dog:.2f}%**")
                    st.progress(int(confidence_dog))
                
                else:
                    # Nếu độ tin cậy đều thấp hơn ngưỡng, kết luận là loài khác
                    max_conf = max(confidence_cat, confidence_dog)
                    st.warning(f"👽 **Kết quả:** AI dự đoán đây là **LOÀI KHÁC** (hoặc vật thể không rõ).")
                    st.info(f"Mô hình không chắc chắn (chỉ đạt {max_conf:.2f}% giống chó/mèo - dưới ngưỡng yêu cầu {THRESHOLD}%)")
                    st.progress(int(max_conf))
                    
            except Exception as e:
                st.error(f"Đã xảy ra lỗi khi phân tích ảnh: {e}")
