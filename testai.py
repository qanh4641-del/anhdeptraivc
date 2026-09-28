import streamlit as st
import pandas as pd

# Tiêu đề trang web
st.title("📊 Web Dashboard - Dữ liệu từ Google Drive")

# Hàm tải dữ liệu từ Drive (Sử dụng cache để không phải tải lại liên tục)
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
    
    # Hiển thị bảng dữ liệu
    st.write("### Xem trước dữ liệu:")
    st.dataframe(df.head(10)) # Hiển thị 10 dòng đầu
    
    # Thêm một chút tương tác Streamlit (Ví dụ: Thống kê mô tả)
    if st.checkbox("Hiển thị tóm tắt thống kê"):
        st.write(df.describe())
        
except Exception as e:
    st.error(f"Đã xảy ra lỗi khi tải dữ liệu: {e}")
