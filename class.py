import torch
from PIL import Image
from transformers import AutoProcessor, SiglipModel
import faiss
import numpy as np
import os
import argparse
from tqdm import tqdm
from pathlib import Path
import shutil

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 初始化設備
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 載入模型和處理器
processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
model = SiglipModel.from_pretrained("google/siglip-so400m-patch14-384").to(device)

# 提取圖片特徵，先進行resize再提取
def extract_features_siglip(image_path):
    image = resize_image(image_path)
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        image_features = model.get_image_features(**inputs)
    return image_features

# 圖片縮放函式
def resize_image(image_path, max_size=384):
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    if max(image.width, image.height) > max_size:
        if image.width > image.height:
            new_width = max_size
            new_height = int(max_size * image.height / image.width)
        else:
            new_height = max_size
            new_width = int(max_size * image.width / image.height)
        image = image.resize((new_width, new_height), Image.LANCZOS)
    return image

# 遍歷目錄並提取所有圖片特徵
def process_images(image_dir):
    image_paths = list(Path(image_dir).rglob("*.jpg")) + \
                  list(Path(image_dir).rglob("*.jpeg")) + \
                  list(Path(image_dir).rglob("*.png")) + \
                  list(Path(image_dir).rglob("*.webp")) + \
                  list(Path(image_dir).rglob("*.bmp"))
    
    features = []
    valid_image_paths = []
    
    for image_path in tqdm(image_paths, desc="提取特徵中"):
        try:
            feature = extract_features_siglip(image_path).detach().cpu().numpy().astype(np.float32)
            features.append(feature)
            valid_image_paths.append(image_path)
        except Exception as e:
            print(f"無法處理圖片 {image_path}: {e}")
    
    features = np.vstack(features)
    return features, valid_image_paths

# 聚類並移動圖片到相應的目錄
def cluster_and_move_images(image_dir, features, image_paths, class_num=20):
    faiss.normalize_L2(features)
    kmeans = faiss.Kmeans(d=features.shape[1], k=class_num, gpu=True)
    kmeans.train(features)
    _, cluster_indices = kmeans.index.search(features, 1)

    for i, (image_path, cluster_idx) in enumerate(zip(image_paths, cluster_indices)):
        cluster_dir = Path(image_path).parent / f'1_class{cluster_idx[0]}'
        cluster_dir.mkdir(exist_ok=True)
        shutil.move(image_path, cluster_dir / image_path.name)
        print(f"移動圖片 {image_path} 到 {cluster_dir / image_path.name}")

if __name__ == "__main__":
    # 解析命令行參數
    parser = argparse.ArgumentParser(description="圖片聚類工具")
    parser = argparse.ArgumentParser(description="圖片聚類工具")
    parser.add_argument('image_dir', type=str, help='指定圖片所在的目錄')
    parser.add_argument('--class', type=int, default=20, help='要分成幾個類別，預設為20')
    args = parser.parse_args()

    image_dir = args.image_dir.replace('\\', '/')
    class_num = args.class_num

    # 處理圖片並提取特徵
    features, image_paths = process_images(image_dir)

    # 聚類並移動圖片
    cluster_and_move_images(image_dir, features, image_paths, class_num=class_num)
