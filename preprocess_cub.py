import os
import shutil
from PIL import Image
from tqdm import tqdm

def load_cub_data(base_path):
    """Load CUB dataset information from text files."""
    # Load image paths
    images_file = os.path.join(base_path, 'images.txt')
    image_paths = {}
    with open(images_file, 'r') as f:
        for line in f:
            img_id, img_path = line.strip().split()
            image_paths[int(img_id)] = os.path.join(base_path, 'images', img_path)

    # Load bounding boxes
    bbox_file = os.path.join(base_path, 'bounding_boxes.txt')
    bboxes = {}
    with open(bbox_file, 'r') as f:
        for line in f:
            img_id, x, y, width, height = map(float, line.strip().split())
            bboxes[int(img_id)] = (int(x), int(y), int(x + width), int(y + height))

    # Load class labels
    labels_file = os.path.join(base_path, 'image_class_labels.txt')
    labels = {}
    with open(labels_file, 'r') as f:
        for line in f:
            img_id, label = map(int, line.strip().split())
            labels[img_id] = label

    # Load train/test split
    split_file = os.path.join(base_path, 'train_test_split.txt')
    splits = {}
    with open(split_file, 'r') as f:
        for line in f:
            img_id, is_train = map(int, line.strip().split())
            splits[img_id] = is_train

    # Load class names
    classes_file = os.path.join(base_path, 'classes.txt')
    class_names = {}
    with open(classes_file, 'r') as f:
        for line in f:
            class_id, class_name = line.strip().split()
            class_names[int(class_id)] = class_name.replace('.', '_')

    return image_paths, bboxes, labels, splits, class_names

def process_cub_dataset(cub_path, output_base_path, target_size=(224, 224)):
    """Process CUB dataset: crop images and split into train/test."""
    print("Loading CUB dataset information...")
    image_paths, bboxes, labels, splits, class_names = load_cub_data(cub_path)

    # Create output directories
    train_dir = os.path.join(output_base_path, 'train_cropped')
    test_dir = os.path.join(output_base_path, 'test_cropped')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print("Processing images...")
    for img_id in tqdm(image_paths.keys()):
        # Get image information
        img_path = image_paths[img_id]
        bbox = bboxes[img_id]
        label = labels[img_id]
        is_train = splits[img_id]
        class_name = class_names[label]

        # Create class directory name
        class_dir_name = f"{label}_{class_name}"
        
        # Determine output directory based on split
        output_dir = os.path.join(train_dir if is_train else test_dir, class_dir_name)
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Open and crop image
            img = Image.open(img_path)
            img_cropped = img.crop(bbox)
            
            # Resize to target size
            img_resized = img_cropped.resize(target_size, Image.LANCZOS)
            
            # Save processed image
            output_filename = os.path.basename(img_path)
            output_path = os.path.join(output_dir, output_filename)
            img_resized.save(output_path)
        except Exception as e:
            print(f"Error processing image {img_path}: {str(e)}")

def main():
    # Set paths
    cub_path = "./datasets/CUB_200_2011"  # Base path to CUB dataset
    output_path = "./datasets/cub200_cropped"  # Output path for processed images
    
    print("Starting CUB-200-2011 dataset preprocessing...")
    process_cub_dataset(cub_path, output_path)
    print("Dataset preprocessing completed!")

if __name__ == "__main__":
    main()