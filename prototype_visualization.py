import torch
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional
from matplotlib.gridspec import GridSpec
import re

class PrototypeVisualizer:
    def __init__(self, analysis_dir: str, save_dir: str):
        """Initialize the visualizer with directories"""
        self.analysis_dir = Path(analysis_dir)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def create_comparison_grid(self, 
                             test_img_path: str,
                             prototype_paths: List[str],
                             activation_values: List[float],
                             save_path: str,
                             class_name: str):
        """Create a grid visualization comparing test image and prototypes"""
        
        # Create figure with custom layout
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(4, 4, figure=fig)
        
        # Original test image (larger)
        ax_test = fig.add_subplot(gs[0:2, 0:2])
        test_img = plt.imread(test_img_path)
        ax_test.imshow(test_img)
        ax_test.set_title(f"Original Test Image\n{class_name}", pad=10)
        
        # Most activated prototypes with their activation values
        for i, (proto_path, act_val) in enumerate(zip(prototype_paths, activation_values)):
            ax = fig.add_subplot(gs[i//2, 2 + i%2])
            proto_img = plt.imread(proto_path)
            ax.imshow(proto_img)
            ax.set_title(f"Prototype {i+1}\nActivation: {act_val:.3f}", pad=10)
            
        # Remove axes
        for ax in fig.get_axes():
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

    def process_image_results(self, image_dir: Path, class_name: str):
        """Process results for a single image"""
        # Set professional fonts with fallbacks
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['DejaVu Serif', 'Liberation Serif', 'Computer Modern Roman', 'Times New Roman']
        plt.rcParams['font.size'] = 11
        
        # Find all top-1 class prototype directories
        top1_dirs = [d for d in image_dir.iterdir() if d.is_dir() and d.name.startswith("top-1_class_prototypes_class")]
        if not top1_dirs:
            print(f"No top-1 prototype directory found in {image_dir}")
            return []
        
        results = []
        # Process each top-1 directory
        for proto_dir in top1_dirs:
            try:
                # Get all prototype and patch pairs
                pairs = []
                for i in range(1, 11):  # 1 to 10
                    proto_pattern = f"top-{i}_activated_prototype_*.png"
                    patch_pattern = f"most_highly_activated_patch_in_original_img_by_top-{i}_class.png"
                    
                    proto_files = list(proto_dir.glob(proto_pattern))
                    patch_files = list(proto_dir.glob(patch_pattern))
                    
                    if proto_files and patch_files:
                        pairs.append((i, proto_files[0], patch_files[0]))
                
                if not pairs:
                    print(f"No prototype-patch pairs found in {proto_dir}")
                    continue
                    
                # Sort pairs by their number
                pairs.sort()
                
                # Create grid with patches and prototypes side by side
                # Add directory index to filename if multiple directories exist
                dir_index = f"_{top1_dirs.index(proto_dir) + 1}" if len(top1_dirs) > 1 else ""
                save_path = self.save_dir / class_name / f"{image_dir.name}_analysis{dir_index}.png"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                
                n_pairs = len(pairs)
                # Calculate number of rows and columns for a more horizontal layout
                n_cols = min(5, n_pairs)  # Maximum 5 pairs per row (increased from 3)
                n_rows = (n_pairs + n_cols - 1) // n_cols  # Ceiling division
                
                # Create figure with minimal height and adjusted spacing
                fig = plt.figure(figsize=(15, 4.2))  # Slightly increased height for titles
                gs = GridSpec(3, 5, figure=fig, 
                            left=0.005, right=0.995,
                            top=0.95, bottom=0.02,  # Adjusted top margin for titles
                            wspace=0.15,  # Horizontal spacing between subplots
                            hspace=0.1)   # Minimal vertical spacing
                
                # White background
                fig.patch.set_facecolor('white')
                
                # Add first row of pairs (0-4)
                for idx in range(min(5, len(pairs))):
                    i, proto_path, patch_path = pairs[idx]
                    ax = fig.add_subplot(gs[0, idx])
                    # Read and process images
                    patch_img = cv2.imread(str(patch_path))
                    patch_img = cv2.cvtColor(patch_img, cv2.COLOR_BGR2RGB)
                    proto_img = cv2.imread(str(proto_path))
                    proto_img = cv2.cvtColor(proto_img, cv2.COLOR_BGR2RGB)
                    
                    # Create a combined image with more space for larger labels
                    combined_height = max(patch_img.shape[0], proto_img.shape[0]) + 30  # Increased from 20
                    combined_width = patch_img.shape[1] + proto_img.shape[1]
                    combined_img = np.ones((combined_height, combined_width, 3), dtype=np.uint8) * 255
                    
                    # Place images at top
                    y_offset = 0
                    combined_img[y_offset:y_offset + patch_img.shape[0], 
                               0:patch_img.shape[1]] = patch_img
                    
                    combined_img[y_offset:y_offset + proto_img.shape[0], 
                               patch_img.shape[1]:] = proto_img
                    
                    # Add labels with a more reliable font
                    font = cv2.FONT_HERSHEY_TRIPLEX  # More reliable professional font
                    font_scale = 0.6
                    font_color = (50, 50, 50)  # Darker gray
                    thickness = 1
                    
                    # Add "test image" label
                    label1 = "Test Image"
                    label1_size = cv2.getTextSize(label1, font, font_scale, thickness)[0]
                    text_x1 = (patch_img.shape[1] - label1_size[0]) // 2
                    text_y = combined_height - 8
                    cv2.putText(combined_img, label1, (text_x1, text_y), 
                              font, font_scale, font_color, thickness)
                    
                    # Add "prototype" label
                    label2 = "Prototype"
                    label2_size = cv2.getTextSize(label2, font, font_scale, thickness)[0]
                    text_x2 = patch_img.shape[1] + (proto_img.shape[1] - label2_size[0]) // 2
                    cv2.putText(combined_img, label2, (text_x2, text_y), 
                              font, font_scale, font_color, thickness)
                    
                    # Display combined image
                    ax.imshow(combined_img)
                    ax.set_title(f"Top-{i}", pad=2, fontsize=11, 
                               fontweight='normal')  # Removed fontfamily
                    ax.set_xticks([])
                    ax.set_yticks([])
                
                # Add class name in middle row with minimal height
                ax_text = fig.add_subplot(gs[1, :])
                ax_text.text(0.5, 0.5, class_name, 
                           horizontalalignment='center',
                           verticalalignment='center',
                           fontsize=14,
                           fontweight='normal')  # Removed fontfamily
                ax_text.set_xticks([])
                ax_text.set_yticks([])
                ax_text.set_frame_on(False)
                
                # Adjust height of middle row to be smaller
                box = ax_text.get_position()
                ax_text.set_position([box.x0, box.y0 + 0.05, box.width, box.height * 0.4])
                
                # Add second row of pairs (5-9)
                for idx in range(5, len(pairs)):
                    i, proto_path, patch_path = pairs[idx]
                    ax = fig.add_subplot(gs[2, idx-5])
                    # Read and process images
                    patch_img = cv2.imread(str(patch_path))
                    patch_img = cv2.cvtColor(patch_img, cv2.COLOR_BGR2RGB)
                    proto_img = cv2.imread(str(proto_path))
                    proto_img = cv2.cvtColor(proto_img, cv2.COLOR_BGR2RGB)
                    
                    # Create a combined image with more space for larger labels
                    combined_height = max(patch_img.shape[0], proto_img.shape[0]) + 30  # Increased from 20
                    combined_width = patch_img.shape[1] + proto_img.shape[1]
                    combined_img = np.ones((combined_height, combined_width, 3), dtype=np.uint8) * 255
                    
                    # Place images at top
                    y_offset = 0
                    combined_img[y_offset:y_offset + patch_img.shape[0], 
                               0:patch_img.shape[1]] = patch_img
                    
                    combined_img[y_offset:y_offset + proto_img.shape[0], 
                               patch_img.shape[1]:] = proto_img
                    
                    # Add labels with a more reliable font
                    font = cv2.FONT_HERSHEY_TRIPLEX  # More reliable professional font
                    font_scale = 0.6
                    font_color = (50, 50, 50)  # Darker gray
                    thickness = 1
                    
                    # Add "test image" label
                    label1 = "Test Image"
                    label1_size = cv2.getTextSize(label1, font, font_scale, thickness)[0]
                    text_x1 = (patch_img.shape[1] - label1_size[0]) // 2
                    text_y = combined_height - 8
                    cv2.putText(combined_img, label1, (text_x1, text_y), 
                              font, font_scale, font_color, thickness)
                    
                    # Add "prototype" label
                    label2 = "Prototype"
                    label2_size = cv2.getTextSize(label2, font, font_scale, thickness)[0]
                    text_x2 = patch_img.shape[1] + (proto_img.shape[1] - label2_size[0]) // 2
                    cv2.putText(combined_img, label2, (text_x2, text_y), 
                              font, font_scale, font_color, thickness)
                    
                    # Display combined image
                    ax.imshow(combined_img)
                    ax.set_title(f"Top-{i}", pad=2, fontsize=11, 
                               fontweight='normal')  # Removed fontfamily
                    ax.set_xticks([])
                    ax.set_yticks([])
                
                # Adjust subplot spacing
                plt.subplots_adjust(hspace=0.1)  # Minimal vertical space between rows
                
                # Save with minimal margins
                plt.savefig(str(save_path), 
                          bbox_inches='tight', 
                          dpi=300,
                          format='png',
                          facecolor='white',
                          edgecolor='none',
                          pad_inches=0.01)
                plt.close()
                
                results.append({
                    "class": class_name,
                    "image_name": image_dir.name,
                    "visualization_path": str(save_path)
                })
            except Exception as e:
                print(f"Error processing directory {proto_dir}: {e}")
                continue
            finally:
                plt.close('all')  # Ensure all figures are closed
        
        return results

    def analyze_class(self, class_dir: Path):
        """Process all images for a specific class"""
        class_name = class_dir.name.replace("pins_", "")  # Remove 'pins_' prefix if present
        results = []
        
        print(f"\nProcessing class: {class_name}")
        
        # Process each image directory in the class
        for img_dir in class_dir.iterdir():
            if img_dir.is_dir() and not img_dir.name.startswith('.'):  # Skip hidden directories
                print(f"  Processing image: {img_dir.name}")
                result = self.process_image_results(img_dir, class_name)
                if result:
                    results.extend(result)
        
        return results

    def analyze_all(self):
        """Process all classes in the analysis directory"""
        all_results = []
        
        # Process each class directory
        for class_dir in self.analysis_dir.iterdir():
            if class_dir.is_dir():
                class_results = self.analyze_class(class_dir)
                all_results.extend(class_results)
        
        # Save summary
        summary_path = self.save_dir / "analysis_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Print summary statistics
        classes = set(r["class"] for r in all_results)
        print(f"\nAnalysis Summary:")
        print(f"Total classes processed: {len(classes)}")
        print(f"Total images processed: {len(all_results)}")
        
        # Per-class statistics - simplified to just count images per class
        print("\nPer-class Statistics:")
        for class_name in sorted(classes):
            class_results = [r for r in all_results if r["class"] == class_name]
            print(f"{class_name}:")
            print(f"  Images processed: {len(class_results)}")
        
        print(f"\nResults saved to: {self.save_dir}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Prototype Analysis Visualization')
    parser.add_argument('--analysis_dir', type=str, 
                       default='app/analysis_results',
                       help='Directory containing analysis results')
    parser.add_argument('--save_dir', type=str, 
                       default='app/visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--class_name', type=str,
                       help='Process only this specific class (optional)')
    
    args = parser.parse_args()
    
    visualizer = PrototypeVisualizer(args.analysis_dir, args.save_dir)
    
    if args.class_name:
        # Get the image directory path
        image_dir = Path(args.analysis_dir)
        
        # Get the class name from the parent directory if not provided
        if not args.class_name:
            args.class_name = image_dir.parent.name.replace('pins_', '')
            
        # Process the single image
        if image_dir.exists():
            result = visualizer.process_image_results(image_dir, args.class_name)
            if result:
                # Save image-specific summary
                summary_path = Path(args.save_dir) / args.class_name / "analysis_summary.json"
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                with open(summary_path, 'w') as f:
                    json.dump(result, f, indent=2)
        else:
            print(f"Image directory not found: {image_dir}")
    else:
        # Process all classes
        visualizer.analyze_all()

if __name__ == '__main__':
    main() 