import os
import shutil
from pathlib import Path
import random
from PIL import Image

def convert_classification_to_yolo(source_dir, output_dir, train_ratio=0.8):
    """
    Chuyển đổi dataset phân loại thành định dạng YOLOv8
    
    Cấu trúc nguồn (source_dir):
    - dataset/
      - anthit/
        - image1.jpg
        - ...
      - anco/
        - image1.jpg
        - ...
    
    Cấu trúc đích (output_dir):
    - yolo_dataset/
      - train/
        - images/
        - labels/
      - val/
        - images/
        - labels/
    """
    # Tạo cấu trúc thư mục YOLOv8
    train_img_dir = os.path.join(output_dir, 'train', 'images')
    train_lbl_dir = os.path.join(output_dir, 'train', 'labels')
    val_img_dir = os.path.join(output_dir, 'val', 'images')
    val_lbl_dir = os.path.join(output_dir, 'val', 'labels')
    
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)
    
    # Xác định các lớp
    classes = {'anco': 0, 'anthit': 1}
    
    # Xử lý từng lớp
    for class_name, class_id in classes.items():
        class_dir = os.path.join(source_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Thư mục {class_dir} không tồn tại, bỏ qua.")
            continue
        
        # Liệt kê tất cả ảnh
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(list(Path(class_dir).glob(ext)))
        
        # Xáo trộn ảnh
        random.shuffle(image_files)
        
        # Phân chia tập train/val
        split_idx = int(len(image_files) * train_ratio)
        train_images = image_files[:split_idx]
        val_images = image_files[split_idx:]
        
        # Xử lý tập train
        for img_path in train_images:
            try:
                # Mở ảnh để lấy kích thước
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
                
                # Copy ảnh sang thư mục train
                shutil.copy(img_path, os.path.join(train_img_dir, img_path.name))
                
                # Tạo file nhãn YOLOv8 (với giả định đối tượng chiếm ~80% kích thước ảnh và nằm ở giữa)
                with open(os.path.join(train_lbl_dir, f"{img_path.stem}.txt"), 'w') as f:
                    # Format: <class_id> <x_center> <y_center> <width> <height>
                    f.write(f"{class_id} 0.5 0.5 0.8 0.8\n")
            except Exception as e:
                print(f"Lỗi xử lý ảnh {img_path}: {e}")
        
        # Xử lý tập val
        for img_path in val_images:
            try:
                # Mở ảnh để lấy kích thước
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
                
                # Copy ảnh sang thư mục val
                shutil.copy(img_path, os.path.join(val_img_dir, img_path.name))
                
                # Tạo file nhãn YOLOv8
                with open(os.path.join(val_lbl_dir, f"{img_path.stem}.txt"), 'w') as f:
                    f.write(f"{class_id} 0.5 0.5 0.8 0.8\n")
            except Exception as e:
                print(f"Lỗi xử lý ảnh {img_path}: {e}")
    
    # Tạo file data.yaml cho YOLOv8
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"""
path: {os.path.abspath(output_dir)}
train: train/images
val: val/images

nc: 2
names: ['anco', 'anthit']
        """)
    
    print(f"Đã chuyển đổi dataset sang định dạng YOLO tại: {output_dir}")
    print(f"File cấu hình: {yaml_path}")
    
    return yaml_path

# Sử dụng hàm
if __name__ == "__main__":
    # Đường dẫn đến thư mục dataset hiện tại của bạn
    source_dataset = "dataset"
    
    # Thư mục đầu ra cho định dạng YOLO
    output_dataset = "yolo_dataset"
    
    # Chuyển đổi dataset
    yaml_file = convert_classification_to_yolo(source_dataset, output_dataset)