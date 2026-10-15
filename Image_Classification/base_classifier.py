import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

tf.get_logger().setLevel('ERROR')

model = MobileNetV2(weights="imagenet")

# Get the last convolutional layer
last_conv_layer_name = "out_relu"

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_gradcam(image_path, heatmap, output_path, alpha=0.4):
    # Load the original image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Rescale heatmap to 0-255
    heatmap = np.uint8(255 * heatmap)
    
    # Apply colormap
    jet = plt.colormaps.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    
    # Create an image with RGB colorized heatmap
    jet_heatmap = cv2.resize(jet_heatmap, (img.shape[1], img.shape[0]))
    jet_heatmap = (jet_heatmap * 255).astype(np.uint8)
    
    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    # Save the superimposed image
    plt.imsave(output_path, superimposed_img.astype(np.uint8))
    return output_path

def classify_image(image_path, output_path="gradcam_result.jpg"):
    try:
        # Load and preprocess image
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # Make predictions
        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        top_pred_index = np.argmax(predictions[0])

        print("\nTop-3 Predictions for", image_path)
        for i, (_, label, score) in enumerate(decoded_predictions):
            print(f"  {i + 1}: {label} ({score:.2f})")

        # Generate heatmap
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=top_pred_index)
        
        # Save GradCAM visualization
        save_gradcam(image_path, heatmap, output_path)
        print(f"\nGradCAM heatmap saved to: {output_path}")

    except Exception as e:
        print(f"Error processing '{image_path}': {e}")

if __name__ == "__main__":
    print("Image Classifier with Grad-CAM (type 'exit' to quit)\n")
    while True:
        image_path = input("Enter image filename: ").strip()
        if image_path.lower() == "exit":
            print("Goodbye!")
            break
        if image_path:
            # Generate output filename for the Grad-CAM visualization
            base, ext = os.path.splitext(image_path)
            output_file = f"{base}_gradcam{ext}"
            classify_image(image_path, output_file)