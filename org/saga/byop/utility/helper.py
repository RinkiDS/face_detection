import os
import cv2
import matplotlib.pyplot as plt
from collections import Counter

class helper:

  def load_images_from_dir(directory):
    path = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust file extensions as needed
            filepath = os.path.join(directory, filename)
            path.append(filepath)
    return path

  def plot_comparing_images(image_to_verify,reference_image):
      img1 = cv2.imread(image_to_verify)
      img2 = cv2.imread(reference_image)
      plt.imshow(img1)
      plt.axis('off')  # Hide axis
      plt.show()
      plt.imshow(img2)
      plt.axis('off')  # Hide axis
      plt.show()

  def likelihood_estimator(data_array,value1,value2):
      # Count occurrences of each value
      counter = Counter(data_array)
      # Compare counts
      #Find the value with the maximum count
      most_common_value, count = counter.most_common(1)[0]
      print(f"The value that occurs the most is: {most_common_value} with {count} occurrences.")
