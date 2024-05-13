import os
import cv2
from PIL import Image 
import numpy as np 
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
from tensorflow import keras
import segmentation_models as sm
from matplotlib import pyplot as plt
import random

from keras import backend as K
import tensorflow.keras.backend as k
from keras.models import load_model 

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
    # Save the predicted image to a file
    predicted_image_path = "uploaded_images/predicted_image.jpg"  # Define the file path where you want to save the image
    predicted_image_pil.save(predicted_image_path)

    return predicted_image_path

def process_input_image(image_source):
  image = np.expand_dims(image_source, 0)

  prediction = satellite_model.predict(image)
  predicted_image = np.argmax(prediction, axis=3)

  predicted_image = predicted_image[0,:,:]
  predicted_image = predicted_image * 50
  return 'Predicted Masked Image', predicted_image