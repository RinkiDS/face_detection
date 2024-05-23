from utility.image_validator import  image_validator
from utility.helper import  helper
import logging

class main:
    # Set up logging
    def __init__(self):
        # log_setup.log_setup()
        # Example usage
        self.logger = logging.getLogger(__name__)

    def run(self):
        directory_path = "/content/drive/MyDrive/data"
          # reading the captured images
        image_paths = helper.load_images_from_dir(directory_path)
        print(image_paths)

        # 1- getting 10 images from frames
        # image_capturing.image_capturing.start_capturing_images_from_vcam(directory_path,10)

        # 2- DO SPOOF CHECK
        most_common_value,count = image_validator.image_spoof_check(image_paths)
        print(f"The value that occurs the most is: {most_common_value} with {count} occurrences.")

        # 3-DO SIMILARITY CHECK
        # image_to_verify = '/content/drive/MyDrive/data/A1.jpg'
        most_common_value,count=image_validator.image_similarity_check(image_paths)
        print(f"The value that occurs the most is: {most_common_value} with {count} occurrences.")




m = main()
m.run()





