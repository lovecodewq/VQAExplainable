#!/usr/bin/env python3
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from PIL import Image

# Add parent directory to path to import dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.vg import vg

def get_distinct_colors(n):
    """Generate n visually distinct colors"""
    base_colors = list(mcolors.TABLEAU_COLORS.values())
    if n <= len(base_colors):
        return base_colors[:n]
    
    # If we need more colors than available, use HSV color space
    return [plt.cm.hsv(i/n) for i in range(n)]

def visualize_image_with_boxes(dataset, idx):
    """
    Visualize an image with its bounding boxes and attributes
    """
    # Get image path and load image
    img_path = dataset.image_path_at(idx)
    img = Image.open(img_path)
    print(f"img_path: {img_path}\nsize: {img.size}")
    
    # Get ground truth data
    gt_data = dataset.gt_roidb()[idx]
    
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(12, 8))
    
    # Display the image
    ax.imshow(img)
    
    # Get the dense attribute matrix
    attr_matrix = gt_data['gt_attributes'].toarray()
    
    # Generate distinct colors for bounding boxes
    colors = get_distinct_colors(len(gt_data['boxes']))
    
    # Plot each bounding box
    for i, (bbox, color) in enumerate(zip(gt_data['boxes'], colors)):
        # Convert bbox coordinates to regular integers to avoid uint16 overflow
        x1, y1, x2, y2 = map(int, bbox)
        w = x2 - x1
        h = y2 - y1
        
        # Get class name and color based on class
        class_idx = gt_data['gt_classes'][i]
        class_name = dataset._classes[class_idx]
        # Use red for background class, original color for others
        box_color = 'red' if class_idx == 0 else color

        # Get attributes for this object
        attr_indices = list(attr_matrix[i])
        attr_names = [dataset._attributes[idx] for idx in attr_indices if idx > 0]
        if not attr_names:
            attr_str = ''
            continue # skip if no attributes
        else:
            attr_str = ', '.join(attr_names)
        # Add label with smart positioning
        label = f"{class_name}\n{attr_str}"
        if y1 > 20:
            text_y = max(0, y1 - 10)
        else:
            text_y = min(img.size[1], y1 + h + 10)
        # Create a Rectangle patch
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor=box_color, facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, text_y, label, color=box_color, fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7))
    
    # Set title
    plt.title(f"Image {idx} with Objects and Attributes\nRed boxes indicate background class")
    
    # Turn off axes
    plt.axis('off')
    
    # Save the plot
    output_dir = 'visualization_output'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'vg_visualization_{idx}.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"\nSaved visualization to: {output_path}")
    plt.close()

def analyze_classes(dataset, gt_data):
    """Analyze class distribution in the ground truth data"""
    print("\nClass Analysis:")
    print(f"Total objects: {len(gt_data['gt_classes'])}")
    
    # Count occurrences of each class
    class_counts = {}
    for class_idx in gt_data['gt_classes']:
        class_name = dataset._classes[class_idx]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Print class distribution
    print("\nClass distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} objects")
    
    # Detailed analysis of background boxes
    background_boxes = np.where(gt_data['gt_classes'] == 0)[0]
    if len(background_boxes) > 0:
        print(f"\nWARNING: Found {len(background_boxes)} boxes with background class!")
        print("\nDetailed analysis of background boxes:")
        for idx in background_boxes:
            box = gt_data['boxes'][idx]
            print(f"\nBackground box {idx}:")
            print(f"  Coordinates: x1={box[0]}, y1={box[1]}, x2={box[2]}, y2={box[3]}")
            print(f"  Box size: width={box[2]-box[0]}, height={box[3]-box[1]}")
            
            # Check overlaps matrix for this box
            overlaps = gt_data['gt_overlaps'][idx].toarray()[0]
            if np.any(overlaps > 0):
                overlap_classes = np.where(overlaps > 0)[0]
                print(f"  Overlaps with classes: {[dataset._classes[i] for i in overlap_classes]}")
            else:
                print("  No overlaps with any class")
            
            # Check attributes
            attrs = gt_data['gt_attributes'][idx].toarray()[0]
            attr_indices = np.nonzero(attrs)[0]
            if len(attr_indices) > 0:
                print(f"  Attributes: {[dataset._attributes[i] for i in attr_indices]}")
            else:
                print("  No attributes")
    else:
        print("\nConfirmed: No boxes are labeled as background class")

def main():
    # Initialize dataset
    print("Initializing Visual Genome dataset...")
    dataset = vg('1600-400-20', 'minitrain')
    
    # Print class information
    print("\nClass Information:")
    print(f"Total classes (including background): {len(dataset._classes)}")
    print(f"Background class index: {dataset._class_to_ind['__background__']}")
    print("\nFirst 10 object classes:")
    for i, class_name in enumerate(dataset._classes[:11]):
        print(f"  {i}: {class_name}")
    
    # Get total number of images
    num_images = len(dataset.image_index)
    print(f"\nTotal number of images: {num_images}")
    
    # Select a random image
    random_idx = random.randint(0, num_images - 1)
    print(f"\nSelected random image index: {random_idx}")
    
    # Load and analyze ground truth for the selected image
    gt_data = dataset.gt_roidb()[random_idx]
    analyze_classes(dataset, gt_data)
    
    # Visualize the image
    print("\nVisualizing image with bounding boxes and attributes...")
    visualize_image_with_boxes(dataset, random_idx)
    print("Visualization complete!")

if __name__ == '__main__':
    main()



