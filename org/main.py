import tensorflow as tf
from keras.models import load_model
from org.saga.byop.utility import spoof_evaluator
from org.saga.byop.utility import image_capturing
from org.saga.byop.utility import log_setup
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import logging
class main:
    # Set up logging
    def __init__(self):
        log_setup.log_setup()
        # Example usage
        self.logger = logging.getLogger(__name__)


    def load_model(model_path):
        model = load_model(model_path)
        print(model)


    def run(self):
        self.logger.debug('This is a debug message')
        self.logger.info('This is an info message')
        self.logger.warning('This is a warning message')
        self.logger.error('This is an error message')
        self.logger.critical('This is a critical message')

        directory_path="C:\\inputFolder"
        #1- getting 10 images from frames
        image_capturing.image_capturing.start_capturing_images_from_vcam(directory_path,10)

        # reading the captured images
        self.get_image_files_from_dir(directory_path)

       # 2-load the model
        model_path = 'C:\\Users\\Rinki\\Downloads\\finalized_model-17may2024 (1).h5'
        model= model = tf.keras.models.load_model(model_path)

       # 3-sending the model to evaluate the image
        evaluator =  spoof_evaluator(model)
        image_path='C:\\\Rinki\\Downloads\\archive\\LCC_FASD\\LCC_FASD_evaluation\\spoof\\spoof_980.png'

       # 4-predict image
        label_name, confidence = evaluator.predict_image_label(image_path)
        print("Prediction for the image:", label_name)
        print("Confidence:", confidence)

    def get_image_files_from_dir(self, directory_path):
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                print(f'Processing file: {filename}')




