import os

import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance

def apply_blackwhite_filter(image_path, output_path="bw_image.png"):
    try:
        # Open the image
        img = Image.open(image_path).convert('RGB')
        img_resized = img.resize((256, 256))
        
        # Convert to grayscale
        img_gray = img_resized.convert('L')
        
        # Slightly increase contrast for a more dramatic black and white look
        enhancer = ImageEnhance.Contrast(img_gray)
        img_bw = enhancer.enhance(1.5)
        
        # Convert back to RGB for matplotlib compatibility
        img_bw = img_bw.convert('RGB')
        
        # Save the result
        plt.imshow(img_bw)
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        print(f"Processed image saved as '{output_path}'.")
        
    except Exception as e:
        print(f"Error processing image: {e}")

def main():
    print("Black & White Filter Processor (type 'exit' to quit)\n")
    while True:
        image_path = input("Enter image filename (or 'exit' to quit): ").strip()
        if image_path.lower() == 'exit':
            print("Goodbye!")
            break
        if not os.path.isfile(image_path):
            print(f"File not found: {image_path}")
            continue
        # derive output filename
        base, ext = os.path.splitext(image_path)
        output_file = f"{base}_bw{ext}"
        apply_blackwhite_filter(image_path, output_file)

if __name__ == "__main__":
    main()
