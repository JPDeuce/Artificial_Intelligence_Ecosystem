# Final Report: Image Classification and Processing

**GitHub Repository:** https://github.com/JPDeuce/Artificial_Intelligence_Ecosystem/tree/main/Image_Classification

## Part 1: Using the Basic Classifier and Implementing Grad-CAM

### Part 1-3: Run the Basic Classifier

#### AI explanation of `base_classifier.py` (Line by Line)

1. `import tensorflow as tf` - Imports TensorFlow library
2. `tf.get_logger().setLevel('ERROR')` - Suppresses verbose TensorFlow logging messages
3. `from tensorflow.keras.applications import MobileNetV2` - Imports the pre-trained MobileNetV2 model
4. `from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions` - Imports preprocessing and prediction decoding functions specific to MobileNetV2
5. `from tensorflow.keras.preprocessing import image` - Imports Keras image utilities for loading images
6. `import numpy as np` - Imports NumPy for numerical operations

7. `model = MobileNetV2(weights="imagenet")` - Loads MobileNetV2 pre-trained on ImageNet dataset (downloads weights on first run)

8. `def classify_image(image_path):` - Defines the function to classify an image
9. `    try:` - Begins error handling
10. `        img = image.load_img(image_path, target_size=(224, 224))` - Loads image and resizes to 224x224 pixels (required input size for MobileNetV2)
11. `        img_array = image.img_to_array(img)` - Converts PIL image to NumPy array (shape: 224x224x3)
12. `        img_array = preprocess_input(img_array)` - Preprocesses pixel values according to MobileNetV2's training requirements
13. `        img_array = np.expand_dims(img_array, axis=0)` - Adds batch dimension (shape: 1x224x224x3)
14. 
15. `        predictions = model.predict(img_array)` - Runs image through model to get raw prediction scores for 1000 ImageNet classes
16. `        decoded_predictions = decode_predictions(predictions, top=3)[0]` - Converts predictions to human-readable labels and confidence scores, returns top 3
17. 
18. `        print("\nTop-3 Predictions for", image_path)` - Prints results header
19. `        for i, (_, label, score) in enumerate(decoded_predictions):` - Iterates through top 3 predictions
20. `            print(f"  {i + 1}: {label} ({score:.2f})")` - Prints each prediction with rank, label, and confidence score
21. `    except Exception as e:` - Catches any errors
22. `        print(f"Error processing '{image_path}': {e}")` - Prints error message
23. 
24. `if __name__ == "__main__":` - Runs only when script is executed directly
25. `    print("Image Classifier (type 'exit' to quit)\n")` - Prints welcome message
26. `    while True:` - Creates interactive loop
27. `        image_path = input("Enter image filename: ").strip()` - Prompts for image filename
28. `        if image_path.lower() == "exit":` - Checks for exit command
29. `            print("Goodbye!")` - Prints goodbye
30. `            break` - Exits loop
31. `        classify_image(image_path)` - Calls classifier on the provided image

*Reflection*: The line-by-line explanation of the code was clear and easy to understand. I have written a few different projects with Python in the past, so my own understanding of the code was pretty accurate as well. I chose an old image of my dog from when he was a puppy, and I was pretty impressed with the results. I figured it would recognize it was a dog or a puppy, but I wasn't expecting it to get the correct breed (Belgian Malinois) with such high confidence!

#### Results for `pup.jpg`

- **Prediction 1**: malinois (0.88) - 88% confidence
- **Prediction 2**: Norwegian_elkhound (0.01) - 1% confidence
- **Prediction 3**: kelpie (0.00) - 0% confidence

### Part 1-4: Implement Grad-CAM

With the help of AI, I integrated Grad-CAM functionality into `base_classifier.py`. To do this, I added the necessary imports (matplotlib, cv2) and implemented functions to generate and save the heatmap. I used TensorFlow's GradientTape to compute the gradients of the predicted class with respect to the feature maps in the last convolutional layer (out_relu for MobileNetV2). Then I computed importance weights by averaging the gradients, multiplied them with the feature maps, applied ReLU, and resized the result to overlay onto the original image. The heatmap was saved as `pup_gradcam.jpg`.

### Part 1-5: Understand Grad-CAM

AI explanation of Grad-CAM:

Grad-CAM (Gradient-weighted Class Activation Mapping) is a technique that helps us see what parts of an image a CNN focuses on when making a prediction. Here's how it works:

1. **Forward Pass** - The image goes through the network to get predictions.
2. **Choose Target Class** - We pick the class we want to explain (in this case, the top prediction).
3. **Compute Gradients** - Using backpropagation, we calculate gradients of that class score with respect to the feature maps of the last convolutional layer.
4. **Get Importance Weights** - We average those gradients across each feature map channel to get weights showing how important each channel is.
5. **Create the Heatmap** - We multiply each feature map by its weight, sum them up, and apply ReLU to keep only positive contributions.
6. **Normalize and Overlay** - We resize the heatmap to match the original image size, normalize it, apply a colormap, and blend it with the original image.

Grad-CAM uses the last convolutional layer because it balances spatial information (where things are) with semantic information (what features mean).

### Part 1-6: Analyze the Heatmap

*My Observations*:

Seeing the explanation of how Grad-CAM works and then looking at the heatmap on my image helped me understand how the classifier was able to identify it as a Belgian Malinois. The strongest highlighted areas were around the face, ears, and upper body, while much of the background received less attention. These areas contain features that can help distinguish the breed, which suggests the model was focusing on relevant parts of the dog when making its prediction. The heatmap gives me a better idea of which areas contributed to the classification, although it doesn't prove exactly how the model made its decision.

## Part 2: Creating and Experimenting with Image Filters

### Part 2-1: Understand Image Filters

#### AI explanation of `basic_filter.py` (Line by Line)

1. `from PIL import Image, ImageFilter` - Imports PIL (Pillow) for image processing and ImageFilter for built-in filters like GaussianBlur
2. `import matplotlib.pyplot as plt` - Imports matplotlib for displaying and saving images
3. `import os` - Imports os module for file operations (checking if file exists, splitting filenames)
4. 
5. `def apply_blur_filter(image_path, output_path="blurred_image.png"):` - Defines function that applies blur filter to an image
6. `    try:` - Begins error handling
7. `        img = Image.open(image_path)` - Opens the image file using PIL
8. `        img_resized = img.resize((128, 128))` - Resizes image to 128x128 pixels (small size for quick processing)
9. `        img_blurred = img_resized.filter(ImageFilter.GaussianBlur(radius=2))` - Applies Gaussian blur with radius 2
10. 
11. `        plt.imshow(img_blurred)` - Displays the blurred image using matplotlib
12. `        plt.axis('off')` - Removes axes from the displayed plot
13. `        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)` - Saves the plot to file with tight bounding box
14. `        plt.close()` - Closes the plot to free memory
15. `        print(f"Processed image saved as '{output_path}'.")` - Prints success message
16. 
17. `    except Exception as e:` - Catches any errors
18. `        print(f"Error processing image: {e}")` - Prints error message
19. 
20. `if __name__ == "__main__":` - Runs only when script is executed directly
21. `    print("Image Blur Processor (type 'exit' to quit)\n")` - Prints welcome message
22. `    while True:` - Creates interactive loop
23. `        image_path = input("Enter image filename (or 'exit' to quit): ").strip()` - Prompts for filename
24. `        if image_path.lower() == 'exit':` - Checks for exit command
25. `            print("Goodbye!")` - Prints goodbye
26. `            break` - Exits loop
27. `        if not os.path.isfile(image_path):` - Checks if file exists
28. `            print(f"File not found: {image_path}")` - Prints error if file not found
29. `            continue` - Goes to next iteration
30. `        # derive output filename` - Comment explaining next lines
31. `        base, ext = os.path.splitext(image_path)` - Splits filename into base and extension
32. `        output_file = f"{base}_blurred{ext}"` - Creates output filename by appending _blurred
33. `        apply_blur_filter(image_path, output_file)` - Applies filter and saves output

*Reflection*: Looking at basic_filter.py, I can see it's applying a Gaussian blur. The code resizes the image to 128x128 first, then applies ImageFilter.GaussianBlur with radius 2. It uses matplotlib to display and save the result without axes. The output filename is created by adding "_blurred" before the file extension. Gaussian blur works by using a Gaussian weighted kernel to average pixels with their neighbors, which smooths out sharp transitions and creates a softer, blurred look. I think the resize to 128x128 is to make processing faster and the blur more noticeable.

### Part 2-2: Design Your Own Artistic Filter

For my filter, I decided to create a black and white filter. I modified basic_filter.py to create new_filter_bw.py. The filter converts the image to grayscale using PIL's convert('L'), then increases the contrast slightly with ImageEnhance.Contrast to make the blacks darker and whites brighter. This gives it a more classic, high-contrast black and white look rather than a flat gray image. I experimented with the contrast adjustment to find a balance that made the dog's facial features and fur stand out without losing too much detail in the darker areas.

**Sample output**: `pup_bw.jpg`

*Description of the effect*: The filter removes all color and creates a classic black and white version of the image. By increasing the contrast, it brings out more definition between light and dark areas, making the fur texture and facial features stand out more distinctly. I like how it gives the photo a timeless, classic feel.

## Reflection on Working with AI

Working with AI to explain code was helpful because it broke things down line by line, which made it easier to understand. For the Grad-CAM part especially, having it explained step by step helped me understand what the algorithm was doing and why. It was also helpful being able to ask questions and get clarification. I found it most useful when I used it to explain concepts and help with implementation, rather than just generating everything at once. Since I already had some Python experience, I could compare the explanations with my own understanding and better follow how the code worked.
