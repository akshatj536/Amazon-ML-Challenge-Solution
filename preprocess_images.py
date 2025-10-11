import os
from glob import glob
from PIL import Image
from tqdm import tqdm

# --- Configuration ---
SOURCE_DIR = 'train_images/'
TARGET_DIR = 'train_images_resized/'
TARGET_SIZE = (256, 256)
# Using a high-quality resampling filter
RESAMPLE_FILTER = Image.Resampling.LANCZOS
# JPEG quality for saved images
JPEG_QUALITY = 95

def preprocess_images():
    """
    Resizes all images from the source directory and saves them to the target directory.
    """
    print(f"Starting image preprocessing...")
    print(f"Source directory: {SOURCE_DIR}")
    print(f"Target directory: {TARGET_DIR}")
    print(f"Target size: {TARGET_SIZE}")

    # Create the target directory if it doesn't exist
    os.makedirs(TARGET_DIR, exist_ok=True)
    print(f"Ensured target directory '{TARGET_DIR}' exists.")

    # Find all image files in the source directory
    # Using a set to avoid duplicates if multiple extensions match the same file
    image_paths = set(glob(os.path.join(SOURCE_DIR, '*.jpg')))
    image_paths.update(glob(os.path.join(SOURCE_DIR, '*.jpeg')))
    image_paths.update(glob(os.path.join(SOURCE_DIR, '*.png')))
    
    image_paths = sorted(list(image_paths))
    
    if not image_paths:
        print("Error: No images found in the source directory. Please check the path.")
        return

    print(f"Found {len(image_paths)} images to preprocess.")

    # Process images with a progress bar
    for path in tqdm(image_paths, desc="Resizing images"):
        try:
            # Open image
            img = Image.open(path)

            # Convert to RGB to ensure 3 channels
            img = img.convert('RGB')

            # Resize the image
            resized_img = img.resize(TARGET_SIZE, RESAMPLE_FILTER)

            # Construct the output path
            filename = os.path.basename(path)
            # Ensure the output filename is .jpg
            base, _ = os.path.splitext(filename)
            output_filename = f"{base}.jpg"
            output_path = os.path.join(TARGET_DIR, output_filename)

            # Save the new image as JPEG
            resized_img.save(output_path, 'JPEG', quality=JPEG_QUALITY)

        except Exception as e:
            print(f"\nError processing file {path}: {e}")
            # Optionally, skip the file and continue
            continue
    
    print("\nImage preprocessing complete.")
    print(f"All resized images have been saved to '{TARGET_DIR}'.")

if __name__ == '__main__':
    preprocess_images()
