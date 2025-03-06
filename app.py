import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import os

# Tiêu đề ứng dụng
st.set_page_config(
    page_title="Phân loại động vật ăn cỏ và ăn thịt",
    page_icon="🦁",
    layout="wide"
)


st.markdown("""
<style>
    .stApp {
        background-color: #1a1a1a;
    }
    
    .main {
        padding: 1rem !important;
    }
    
    /* Header Styling */
    .header {
        background: linear-gradient(135deg, #2c2c2c, #1a1a1a);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .title {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .subtitle {
        color: #9e9e9e;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        width: 100%;
    }
    
    [data-testid="stFileUploadDropzone"] {
        background: rgba(34,34,34,0.7) !important;
        border: 2px dashed #4CAF50 !important;
        border-radius: 15px !important;
        padding: 2rem !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #69F0AE !important;
        box-shadow: 0 0 20px rgba(76,175,80,0.2) !important;
    }
    
    /* Result Styling */
    .result-container {
        background: #2c2c2c;
        border-radius: 15px;
        padding: 1rem;
        margin-top: 1rem;
        text-align: center;
        animation: fadeIn 0.5s ease-in;
    }
    
    .result-carnivore {
        border: 3px solid #FF5252;
        background: linear-gradient(135deg, #2c2c2c, #331111);
    }
    
    .result-herbivore {
        border: 3px solid #69F0AE;
        background: linear-gradient(135deg, #2c2c2c, #113322);
    }
    
    .result-text {
        font-size: 1.2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .carnivore-text {
        color: #FF5252;
    }
    
    .herbivore-text {
        color: #69F0AE;
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Image Display */
    .uploaded-image {
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        margin: 1rem 0;
    }
    
    /* Grid Layout */
    .image-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 1rem;
        padding: 1rem;
    }
    
    .image-item {
        background: #2c2c2c;
        border-radius: 15px;
        padding: 1rem;
        animation: fadeIn 0.5s ease-in;
    }
    
    .image-name {
        color: #ffffff;
        font-size: 1rem;
        margin: 0.5rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Tiêu đề và mô tả
st.markdown("""
<div class="header">
    <div class="title">🦁 Phân loại động vật ăn cỏ và ăn thịt 🦌</div>
    <div class="subtitle">Tải lên ảnh động vật để phân loại bằng YOLOv8</div>
</div>
""", unsafe_allow_html=True)

# Tạo sidebar
with st.sidebar:
    st.header("Cài đặt")
    confidence = st.slider(
        "Ngưỡng độ tin cậy", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.2,
        step=0.05
    )
    
    show_labels = st.checkbox("Hiển thị nhãn", value=True)
    show_conf = st.checkbox("Hiển thị độ tin cậy", value=True)
    
    st.markdown("---")
    st.markdown("### Về ứng dụng")
    st.markdown("""
    Ứng dụng này sử dụng mô hình YOLOv8 để phân loại động vật thành:
    - 🌿 **Động vật ăn cỏ**
    - 🥩 **Động vật ăn thịt**
    
    Mô hình có thể phát hiện nhiều động vật trong cùng một ảnh.
    """)

# Tải model
@st.cache_resource
def load_model():
    # Đường dẫn đến mô hình đã huấn luyện
    model_path = "runs\\detect\\herbivore_carnivore_model2\\weights\\best.pt"
    
    if not os.path.exists(model_path):
        st.warning("Không tìm thấy mô hình đã huấn luyện. Đang sử dụng mô hình mặc định...")
        return YOLO("yolov8n.pt")
    
    return YOLO(model_path)

# Tải mô hình
with st.spinner("Đang tải mô hình..."):
    model = load_model()

# Hàm xử lý và hiển thị kết quả
def process_image(img, file_name):
    # Chuyển đổi ảnh sang định dạng numpy
    img_array = np.array(img)
    
    # Dự đoán với YOLOv8
    results = model.predict(
        source=img_array,
        conf=confidence,
        save=False
    )
    
    # Lấy danh sách các bounding boxes
    boxes = results[0].boxes
    
    # Nếu không phát hiện đối tượng nào
    if len(boxes) == 0:
        st.warning(f"Không phát hiện động vật nào trong ảnh với ngưỡng tin cậy {confidence}")
        return

    # Chuyển bounding box thành danh sách để xử lý
    filtered_boxes = []
    seen_classes = {}  # Lưu trữ độ tin cậy cao nhất của mỗi loại động vật
    
    for box in boxes:
        cls_id = int(box.cls[0].item())  # Nhãn của đối tượng
        conf_score = box.conf[0].item()  # Độ tin cậy
        
        # Nếu đã có một bounding box cho đối tượng này, giữ cái có độ tin cậy cao hơn
        if cls_id in seen_classes:
            if conf_score > seen_classes[cls_id]:
                seen_classes[cls_id] = conf_score
                filtered_boxes = [b for b in filtered_boxes if int(b.cls[0].item()) != cls_id]  # Xóa box cũ
                filtered_boxes.append(box)
        else:
            seen_classes[cls_id] = conf_score
            filtered_boxes.append(box)

    # Tạo ảnh với bounding boxes sau khi lọc
    results[0].boxes = filtered_boxes  # Cập nhật bounding boxes
    res_plotted = results[0].plot()
    res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
    
    # Hiển thị kết quả ảnh
    st.image(res_plotted_rgb, caption=f"Kết quả phân tích: {file_name}", use_container_width=True)

    # Hiển thị thông tin về các đối tượng phát hiện được
    st.markdown("### Kết quả phát hiện")
    for box in filtered_boxes:
        cls_id = int(box.cls[0].item())
        conf_score = box.conf[0].item()

        animal_type = "Động vật ăn thịt" if cls_id == 0 else "Động vật ăn cỏ"
        animal_icon = "🥩" if cls_id == 0 else "🌿"
        result_class = "carnivore" if cls_id == 0 else "herbivore"

        # Hiển thị kết quả với styling
        st.markdown(f"""
        <div class="result-container result-{result_class}">
            <div class="result-text {result_class}-text">{animal_icon} {animal_type}</div>
            <div class="confidence-text">Độ tin cậy: {conf_score*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

# Upload ảnh
uploaded_file = st.file_uploader(
    "Tải lên ảnh động vật",
    type=["jpg", "jpeg", "png"],
    help="Hỗ trợ định dạng: JPG, PNG, JPEG"
)

# Xử lý ảnh được tải lên
if uploaded_file is not None:
    try:
        # Đọc ảnh
        image = Image.open(uploaded_file)
        
        # Hiển thị ảnh gốc
        st.markdown("### Ảnh gốc")
        st.image(image, caption=f"Ảnh gốc: {uploaded_file.name}", use_container_width=True)
        
        # Xử lý và hiển thị kết quả
        st.markdown("### Kết quả phân tích")
        process_image(image, uploaded_file.name)
        
    except Exception as e:
        st.error(f"Lỗi khi xử lý ảnh: {str(e)}")

# Demo với ảnh mẫu
st.markdown("---")
st.markdown("### Hoặc thử với ảnh mẫu")

# Tạo layout 2 cột cho ảnh mẫu
col1, col2 = st.columns(2)

with col1:
    if st.button("🦌 Thử với ảnh động vật ăn cỏ"):
        # Đường dẫn đến ảnh mẫu động vật ăn cỏ
        sample_img_path = "1a94573fa2.jpg"
        
        # Kiểm tra nếu file tồn tại
        if os.path.exists(sample_img_path):
            image = Image.open(sample_img_path)
            process_image(image, "Ảnh mẫu động vật ăn cỏ")
        else:
            st.warning("Không tìm thấy ảnh mẫu. Vui lòng tải lên ảnh của bạn.")

with col2:
    if st.button("🦁 Thử với ảnh động vật ăn thịt"):
        # Đường dẫn đến ảnh mẫu động vật ăn thịt
        sample_img_path = "0b54dde5f5.jpg"
        
        # Kiểm tra nếu file tồn tại
        if os.path.exists(sample_img_path):
            image = Image.open(sample_img_path)
            process_image(image, "Ảnh mẫu động vật ăn thịt")
        else:
            st.warning("Không tìm thấy ảnh mẫu. Vui lòng tải lên ảnh của bạn.")
            