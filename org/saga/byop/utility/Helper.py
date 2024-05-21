import os
import cv2
import matplotlib.pyplot as plt

class Helper:

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
      value1_count = data_array.count(value1)
      value2_count = data_array.count(value2)

      # Compare counts
      if value1_count > value2_count:
          print("There are more {value1} values.")
      elif value1_count < value2_count:
          print(f"There are more {value2} values.")
      else:
          print(f"There are an equal number of {value1} and {value2} values: {count_value1} each.")
