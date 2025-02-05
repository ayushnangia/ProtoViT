import os
import Augmentor
from tqdm import tqdm

def augment_training_data(train_dir, output_dir, samples_per_class=5):
    """
    Augment training data using Augmentor.
    Args:
        train_dir: Directory containing training images organized in class folders
        output_dir: Directory to save augmented images
        samples_per_class: Number of augmented samples to generate per original image
    """
    print("Starting data augmentation...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of class directories
    class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    
    for class_dir in tqdm(class_dirs, desc="Processing classes"):
        class_path = os.path.join(train_dir, class_dir)
        output_class_path = os.path.join(output_dir, class_dir)
        
        # Create pipeline
        p = Augmentor.Pipeline(class_path)
        
        # Add operations with probabilities and parameters suitable for face images
        p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
        p.zoom(probability=0.5, min_factor=1.1, max_factor=1.3)
        p.flip_left_right(probability=0.3)  # Reduced probability for face images
        p.random_brightness(probability=0.5, min_factor=0.8, max_factor=1.2)
        p.random_contrast(probability=0.5, min_factor=0.8, max_factor=1.2)
        p.random_distortion(probability=0.3, grid_width=4, grid_height=4, magnitude=2)
        
        # Generate samples
        num_samples = len(os.listdir(class_path)) * samples_per_class
        p.sample(num_samples)
        
        # Move augmented images to the correct output directory
        augmented_dir = os.path.join(class_path, "output")
        if os.path.exists(augmented_dir):
            os.makedirs(output_class_path, exist_ok=True)
            for img in os.listdir(augmented_dir):
                src = os.path.join(augmented_dir, img)
                dst = os.path.join(output_class_path, img)
                os.rename(src, dst)
            os.rmdir(augmented_dir)

def main():
    train_dir = "./datasets/pins/train"
    output_dir = "./datasets/pins/train_augmented"
    
    print("Starting training data augmentation...")
    augment_training_data(train_dir, output_dir)
    print("Data augmentation completed!")

if __name__ == "__main__":
    main() 