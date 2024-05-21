import tensorflow as tf
from keras.models import load_model
from utility import SpoofEvaluator
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class main:

    def load_model(model_path):
        model = load_model(model_path)
        print(model)


    def run(self):

       model_path = 'C:\\Users\\Rinki\\Downloads\\finalized_model-17may2024 (1).h5'
       # load the model
       model= model = tf.keras.models.load_model(model_path)

       # sending the model to evaluate the image
       evaluator =  SpoofEvaluator(model)

       image_path='C:\\\Rinki\\Downloads\\archive\\LCC_FASD\\LCC_FASD_evaluation\\spoof\\spoof_980.png'

       # predict image
       label_name, confidence = evaluator.predict_image_label(image_path)
       print("Prediction for the image:", label_name)
       print("Confidence:", confidence)

       if(label_name=="REAL"):
           print("Real")




