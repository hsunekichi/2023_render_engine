from PIL import Image
import os
import sys

def convert_image_to_24bit_bmp(input_path, output_path):
    try:
        img = Image.open(input_path)

        # If the image mode is "LA," convert it to RGB mode
        if img.mode == "LA":
            img = img.convert("RGB")

        # Convert the image to RGB mode if it's not already
        if img.mode != "RGB":
            img = img.convert("RGB")

        img.save(output_path, "BMP")
        print(f"Image converted to 24-bit BMP: {input_path} -> {output_path}")
    except Exception as e:
        print(f"An error occurred while processing {input_path}: {str(e)}")

def batch_convert_images_to_24bit_bmp(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".tga")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".bmp")
            convert_image_to_24bit_bmp(input_path, output_path)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_images.py input_directory output_directory")
    else:
        input_directory = sys.argv[1]
        output_directory = sys.argv[2]
        batch_convert_images_to_24bit_bmp(input_directory, output_directory)
