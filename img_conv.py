import os
import argparse
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def resize_images(input_dir, output_dir, size=(160, 160)):
    """
    Resize all images in the input directory and save them to the output directory.
    Preserves the folder structure where each subfolder is a class.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Get all class folders (subdirectories)
    class_folders = [d for d in input_path.iterdir() if d.is_dir()]
    
    for class_folder in class_folders:
        class_name = class_folder.name
        output_class_path = output_path / class_name
        output_class_path.mkdir(exist_ok=True)
        
        # Get all image files in the class folder
        image_files = [f for f in class_folder.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']]
        
        print(f"Processing {len(image_files)} images in class '{class_name}'...")
        
        for img_path in tqdm(image_files):
            try:
                with Image.open(img_path) as img:
                    # Convert to RGB if the image has an alpha channel
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')
                    
                    # Resize the image
                    resized_img = img.resize(size, Image.Resampling.LANCZOS)
                    # Apply central crop (87.5% of the image) before resizing
                    width, height = resized_img.size
                    crop_width = int(width * 0.875)
                    crop_height = int(height * 0.875)
                    left = (width - crop_width) // 2
                    top = (height - crop_height) // 2
                    right = left + crop_width
                    bottom = top + crop_height

                    resized_img = resized_img.crop((left, top, right, bottom))
                    resized_img = resized_img.resize(size, Image.Resampling.LANCZOS)
                    # Save the resized image
                    output_img_path = output_class_path / img_path.name
                    resized_img.save(output_img_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resize images to 160x160 pixels')
    parser.add_argument('--input', type=str, required=True, help='Input directory with class subfolders')
    parser.add_argument('--output', type=str, required=True, help='Output directory for resized images')
    
    args = parser.parse_args()
    
    resize_images(args.input, args.output)
    print("Image resizing completed!")