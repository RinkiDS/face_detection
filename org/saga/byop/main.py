import tensorflow as tf
from tensorflow.keras.models import load_model

from utility.spoof_evaluator import spoof_evaluator
from utility import image_capturing
from utility import log_setup
from utility import Helper
from utility import image_similarity_matcher

import logging
class main:
    # Set up logging
    def __init__(self):
        log_setup.log_setup()
        # Example usage
        self.logger = logging.getLogger(__name__)

    def run(self):

       directory_path="/content/drive/MyDrive/data"

       #1- getting 10 images from frames
       #image_capturing.image_capturing.start_capturing_images_from_vcam(directory_path,10)

       # reading the captured images
       Helper.get_image_files_from_dir(directory_path)

       # 2-load the model
       model_path = '/content/drive/MyDrive/data/finalized_model-17may2024.h5'
       model= load_model(model_path)

       # 3-sending the model to evaluate the image
       evaluator =  spoof_evaluator(model)
       image_path='C:\\\Rinki\\Downloads\\archive\\LCC_FASD\\LCC_FASD_evaluation\\spoof\\spoof_980.png'

       # 4-predict image for real or spoof
       label_name, confidence = evaluator.predict_image_label(image_path)
       print("Prediction for the image:", label_name)
       print("Confidence:", confidence)

       # 5-match the image for similarity
       image_to_verify = 'C:\\Users\\Rinki\\Downloads\\A1.jpg'
       reference_image = 'C:\\Users\\Rinki\\Downloads\\B1.jpg'
       Helper.plot_comparing_images(image_to_verify,reference_image)
       image_similarity_matcher(image_to_verify,reference_image)







