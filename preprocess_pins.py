import os
import shutil
import random
from tqdm import tqdm
import cv2
import numpy as np

def create_directory_structure():
    """Create the necessary directory structure for the dataset."""
    base_dirs = ['datasets/pins/train', 'datasets/pins/test']
    for dir_path in base_dirs:
        os.makedirs(dir_path, exist_ok=True)
    return base_dirs[0], base_dirs[1]

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess a single image."""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize image
    img = cv2.resize(img, target_size)
    
    return img

def split_dataset(source_dir, train_dir, test_dir, train_ratio=0.8):
    """Split the dataset into train and test sets."""
    # Get all class directories
    class_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    for class_dir in tqdm(class_dirs, desc="Processing classes"):
        # Create class directories in train and test
        train_class_dir = os.path.join(train_dir, class_dir)
        test_class_dir = os.path.join(test_dir, class_dir)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)
        
        # Get all images in the class
        source_class_dir = os.path.join(source_dir, class_dir)
        images = [f for f in os.listdir(source_class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Shuffle images
        random.shuffle(images)
        
        # Split point
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        # Process and copy train images
        for img_name in train_images:
            src_path = os.path.join(source_class_dir, img_name)
            processed_img = preprocess_image(src_path)
            if processed_img is not None:
                dst_path = os.path.join(train_class_dir, img_name)
                cv2.imwrite(dst_path, cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))
        
        # Process and copy test images
        for img_name in test_images:
            src_path = os.path.join(source_class_dir, img_name)
            processed_img = preprocess_image(src_path)
            if processed_img is not None:
                dst_path = os.path.join(test_class_dir, img_name)
                cv2.imwrite(dst_path, cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Source directory containing the original dataset
    source_dir = "105_classes_pins_dataset"
    
    # Create directory structure
    train_dir, test_dir = create_directory_structure()
    
    print("Starting dataset preprocessing and splitting...")
    split_dataset(source_dir, train_dir, test_dir)
    print("Dataset preprocessing and splitting completed!")

if __name__ == "__main__":
    main() 