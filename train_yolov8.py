from ultralytics import YOLO

def train_yolov8_model(data_yaml, epochs=50, img_size=640, batch_size=8):
    """
    Huấn luyện mô hình YOLOv8 với dataset đã chuyển đổi
    
    Args:
        data_yaml (str): Đường dẫn đến file data.yaml
        epochs (int): Số epoch huấn luyện
        img_size (int): Kích thước ảnh đầu vào (nên để 320, 416, 512 hoặc 640)
        batch_size (int): Kích thước batch (cố gắng để >= 8 nếu có GPU tốt)
    
    Returns:
        Path đến mô hình đã huấn luyện
    """
    # Tải mô hình YOLOv8 pre-trained
    model = YOLO('yolov8n.pt')
    
    # Huấn luyện mô hình
    results = model.train(
        data=data_yaml,          # Đường dẫn đến file data.yaml
        epochs=epochs,           # Số lượng epoch
        imgsz=img_size,          # Kích thước ảnh đầu vào
        batch=batch_size,        # Kích thước batch
        patience=10,             # Dừng sớm nếu mô hình không cải thiện sau 10 epochs
        name='herbivore_carnivore_model' # Tên folder chứa kết quả huấn luyện
    )
    
    # Đường dẫn đến mô hình đã train
    model_path = results.save_dir
    print(f"✅ Huấn luyện xong mô hình YOLOv8. Mô hình được lưu tại: {model_path}")
    
    # Đường dẫn đến file weights tốt nhất
    best_model = f"{model_path}/weights/best.pt"
    
    return best_model

if __name__ == "__main__":
    # Đường dẫn chính xác đến file data.yaml
    yaml_file = "C:\\Users\\admin\\Desktop\\XLA_BTL\\yolo_dataset\\data.yaml"
    
    # Huấn luyện mô hình
    model_path = train_yolov8_model(
        data_yaml=yaml_file,
        epochs=50,               # Số epoch tùy thuộc vào dataset
        img_size=640,            # Kích thước ảnh phù hợp với YOLOv8
        batch_size=8             # Tăng batch size nếu có GPU mạnh
    )
