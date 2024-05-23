import tensorflow as tf
from tensorflow.keras.models import load_model
from utility.spoof_evaluator import spoof_evaluator
from utility.helper import helper
from utility.image_similarity_matcher import image_similarity_matcher

import logging


class main:
    # Set up logging
    def __init__(self):
        # log_setup.log_setup()
        # Example usage
        self.logger = logging.getLogger(__name__)

    def run(self):
        directory_path = "/content/drive/MyDrive/data"

        # 1- getting 10 images from frames
        # image_capturing.image_capturing.start_capturing_images_from_vcam(directory_path,10)

        # reading the captured images
        image_paths = helper.load_images_from_dir(directory_path)

        print(image_paths)

        # 2-load the model
        print("Loading Model");
        model_path = '/content/drive/MyDrive/model/finalized_model-21may2024.h5'
        model = load_model(model_path)
        print("***************Model loaded successfully**********");
        #  3-sending the model to evaluate the image
        evaluator = spoof_evaluator(model)
        print("**************Model Initialized****************")
        image_path = '/content/drive/MyDrive/data/spoof_980.png'

        # 4-predict image for real or spoof
        label_names = evaluator.predict_images_labels(image_paths)
        for label_name in label_names:
            print(label_name)
        helper.likelihood_estimator(label_names, "REAL", "SPOOF")
        # 5-match the image for similarity
        # image_to_verify = '/content/drive/MyDrive/data/A1.jpg'
        results = []
        reference_image = '/content/drive/MyDrive/data/A2.jpg'
        for image_to_verify in image_paths:
            helper.plot_comparing_images(image_to_verify, reference_image)
            result = image_similarity_matcher.image_similarity_match(image_to_verify, reference_image)
            print("Result of Similarity is ", result)
            results.append(result)
        most_common_value=helper.likelihood_estimator(results, "True", "False")


m = main()
m.run()





