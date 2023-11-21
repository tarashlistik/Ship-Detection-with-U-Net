import matplotlib.pyplot as plt
import os
from keras.models import load_model

from utils import valid_x, IoU

# Load the segmentation model from a saved file, providing a custom metric (IoU)
loaded_model = load_model('seg_model.h5', custom_objects={"IoU": IoU})

# Make predictions on the validation dataset using the loaded model
pred_y = loaded_model.predict(valid_x)
print(pred_y.shape, pred_y.min(axis=0).max(), pred_y.max(axis=0).min(), pred_y.mean())

# Create a folder for output images if it doesn't exist
output_folder = 'output_images'
os.makedirs(output_folder, exist_ok=True)

# Visualize a subset of images and their predicted masks
for index in range(6):
    # Extract input image and predicted mask for the current index
    input_image = valid_x[index]
    predicted_mask = pred_y[index]

    # Create a new figure for each image
    plt.figure()

    # Plot the original image
    plt.subplot(1, 2, 1)
    plt.imshow(input_image)
    plt.title('Original Image')

    # Plot the predicted mask
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_mask, cmap='gray')
    plt.title('Predicted Mask')

    # Save the figure with a unique filename in the output folder
    output_path = os.path.join(output_folder, f'image_{index + 1}.png')
    plt.savefig(output_path)

    # Close the figure to release resources
    plt.close()