import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
@st.cache_data
def load_data_from_drive(file_id):
    # Chuyển đổi link Drive thành link tải trực tiếp (direct download link)
    download_url = f"https://drive.google.com/uc?id={file_id}"
    
    # Đọc dữ liệu (Đổi read_csv thành read_excel nếu là file .xlsx)
    df = pd.read_csv(download_url) 
    return df

# Thay thế bằng ID file của bạn ở Bước 1
FILE_ID = "1nAlUWEsaxqOSGEP23nJxn9h3VQYU441L" # <--- THAY ID CỦA BẠN VÀO ĐÂY

st.write("Đang tải dữ liệu từ Google Drive...")

try:
    # Lấy dữ liệu
    df = load_data_from_drive(FILE_ID)
    
    st.success("Tải dữ liệu thành công!")

# Cấu hình trang
st.set_page_config(page_title="AI Nhận diện Chó Mèo", page_icon="🐶", layout="centered")

st.title("🐶🐱 Trợ lý AI Nhận diện Chó & Mèo")
st.markdown("Tải một bức ảnh lên và AI (EfficientNetB7) sẽ cho bạn biết đó là chó hay mèo!")

# Hàm tải mô hình (Dùng cache_resource để load model 1 lần duy nhất, tránh nặng máy)
@st.cache_resource
def load_ai_model():
    # Tên file model phải khớp với file bạn lưu từ Colab
    model_path = 'EfficientNetB7_CatDog.h5'
    model = tf.keras.models.load_model(model_path)
    return model

# Tải model
try:
    with st.spinner("Đang tải mô hình AI, vui lòng đợi vài giây..."):
        model = load_ai_model()
except Exception as e:
    st.error(f"⚠️ KHÔNG TÌM THẤY MÔ HÌNH AI!")
    st.info("Vui lòng tải file 'EfficientNetB7_CatDog.h5' từ Google Drive của bạn và đặt vào cùng thư mục với file app.py này.")
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
