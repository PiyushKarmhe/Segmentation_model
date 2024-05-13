import os
import cv2
from PIL import Image 
import numpy as np 
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
from tensorflow import keras
import segmentation_models as sm

from keras import backend as K
import tensorflow.keras.backend as k
from keras.models import load_model 
import base64
import io
from matplotlib.colors import ListedColormap

def jaccard_coef(y_true, y_pred):
  y_true_flatten = K.flatten(y_true)
  y_pred_flatten = K.flatten(y_pred)
  intersection = K.sum(y_true_flatten * y_pred_flatten)
  final_coef_value = (intersection + 1.0) / (K.sum(y_true_flatten) + K.sum(y_pred_flatten) - intersection + 1.0)
  return final_coef_value

weights = [0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]
dice_loss = sm.losses.DiceLoss(class_weights = weights)
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

satellite_model = load_model('satellite-imagery_new.h5',
                         custom_objects=({'dice_loss_plus_1focal_loss': total_loss, 
                                          'jaccard_coef': jaccard_coef}))

satellite_model.get_config()

def predict(ImagePathUplaoded): 
    # Define user-defined colors for different segments
    segment_colors = {
      0: (0, 255, 0),   # Green
      1: (255, 255, 0), # Yellow
      2: (165, 42, 42), # Brown
      3: (0, 0, 0),     # Black
      4: (255, 165, 0), # Orange
      5: (255, 192, 203) # Pink
    }

    current_directory = os.getcwd()
    print("Current Directory:", current_directory)

    image = Image.open(f"{current_directory}/{ImagePathUplaoded}")
    image = image.resize((256,256))
    image = np.array(image)
    image = np.expand_dims(image, 0)
    print("Uploaded Image : ",image)

    prediction = satellite_model.predict(image)
    predicted_image = np.argmax(prediction, axis=3)
    predicted_image = predicted_image[0,:,:]
    predicted_image = predicted_image * 50
    print("Predicted Image : ",predicted_image)

    predicted_image_pil = Image.fromarray(np.uint8(predicted_image))

    # Create a custom colormap using the user-defined colors
    cmap = ListedColormap([segment_colors[i] for i in range(len(segment_colors))])

    predicted_image_pil = predicted_image_pil.convert('P')
    predicted_image_pil.putpalette([x for rgb in cmap.colors for x in rgb])

    # Save the predicted image to a file
    predicted_image_path = "uploaded_images/predicted_image.png"  # Define the file path where you want to save the image
    predicted_image_pil.save(predicted_image_path)

    image_buffer = io.BytesIO()
    predicted_image_pil.save(image_buffer, format="PNG")
    image_base64 = base64.b64encode(image_buffer.getvalue()).decode('utf-8')

    return image_base64

def process_input_image(image_source):
  image = np.expand_dims(image_source, 0)

  prediction = satellite_model.predict(image)
  predicted_image = np.argmax(prediction, axis=3)

  predicted_image = predicted_image[0,:,:]
  predicted_image = predicted_image * 50
  return 'Predicted Masked Image', predicted_image