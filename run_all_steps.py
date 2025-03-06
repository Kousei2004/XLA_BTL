import os
import subprocess
import shutil

def check_dependencies():
    """Kiểm tra và cài đặt các thư viện cần thiết"""
    try:
        import ultralytics
        print("✓ Đã cài đặt ultralytics")
    except ImportError:
        print("Đang cài đặt ultralytics...")
        subprocess.check_call(["pip", "install", "ultralytics"])
    
    try:
        import streamlit
        print("✓ Đã cài đặt streamlit")
    except ImportError:
        print("Đang cài đặt streamlit...")
        subprocess.check_call(["pip", "install", "streamlit"])

def run_complete_workflow():
    """Chạy toàn bộ quy trình từ chuyển đổi dataset đến triển khai ứng dụng"""
    print("\n=== KIỂM TRA VÀ CÀI ĐẶT THƯ VIỆN ===")
    check_dependencies()
    
    print("\n=== BƯỚC 1: CHUYỂN ĐỔI DATASET ===")
    # Đường dẫn đến dataset gốc của bạn
    dataset_dir = "dataset"
    
    # Kiểm tra xem dataset có tồn tại không
    if not os.path.exists(dataset_dir):
        print(f"Lỗi: Không tìm thấy thư mục dataset tại {os.path.abspath(dataset_dir)}")
        return False
    
    # Kiểm tra cấu trúc dataset
    if not (os.path.exists(os.path.join(dataset_dir, "anco")) and 
            os.path.exists(os.path.join(dataset_dir, "anthit"))):
        print("Lỗi: Dataset phải có cấu trúc: dataset/anco và dataset/anthit")
        return False
    
    # Chạy script chuyển đổi dataset
    from convert_to_yolo_format import convert_classification_to_yolo
    yaml_path = convert_classification_to_yolo(dataset_dir, "yolo_dataset")
    
    print("\n=== BƯỚC 2: HUẤN LUYỆN MÔ HÌNH YOLOV8 ===")
    # Huấn luyện mô hình
    from train_yolov8 import train_yolov8_model
    model_path = train_yolov8_model(yaml_path)
    
    print("\n=== BƯỚC 3: TRIỂN KHAI ỨNG DỤNG STREAMLIT ===")
    # Copy mô hình đã huấn luyện vào thư mục ứng dụng (nếu cần)
    if not os.path.exists(model_path):
        print(f"Lỗi: Không tìm thấy mô hình đã huấn luyện tại {model_path}")
        return False
    
    # Tạo thư mục mẫu cho ảnh demo (nếu có)
    print("Bạn có thể thêm ảnh mẫu vào thư mục hiện tại:")
    print("- sample_herbivore.jpg: Mẫu động vật ăn cỏ")
    print("- sample_carnivore.jpg: Mẫu động vật ăn thịt")
    
    # Chạy ứng dụng Streamlit
    print("\n=== KHỞI ĐỘNG ỨNG DỤNG ===")
    print("Khởi động ứng dụng Streamlit...")
    print("Hãy mở trình duyệt và truy cập địa chỉ hiển thị bên dưới...")
    subprocess.run(["streamlit", "run", "yolo_streamlit_app.py"])
    
    return True

if __name__ == "__main__":
    run_complete_workflow()