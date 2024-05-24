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

  def likelihood_estimator(data_array):
      # Count occurrences of each value
      counter = Counter(data_array)
      # Compare counts
      #Find the value with the maximum count
      most_common_value, count = counter.most_common(1)[0]
      #print(f"The value that occurs the most is: {most_common_value} with {count} occurrences.")
      return  most_common_value,count

  def likelihood_estimator_max_count_and_percentage(arr):
      # Count occurrences of each value in the array
      counts = Counter(arr)

      # Find the maximum count
      max_count = max(counts.values())

      # Find the values with the maximum count
      max_values = [value for value, count in counts.items() if count == max_count]

      # Calculate the percentage of occurrence of the max value
      max_percentage = (max_count / len(arr)) * 100

      # Check if there are multiple values with the same maximum count
      if len(max_values) > 1:
          print("Multiple values with the same maximum count:")
          for value in max_values:
              print(f"Value: {value}, Count: {max_count}, Percentage: {max_percentage}%")
      else:
          max_value = max_values[0]
          print(f"Maximum count value: {max_value}, Count: {max_count}, Percentage: {max_percentage}%")
