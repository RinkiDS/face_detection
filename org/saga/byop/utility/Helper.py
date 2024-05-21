import os
import cv2
import matplotlib.pyplot as plt

def get_image_files_from_dir (directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            print(f'Processing file: {filename}')

def plot_comparing_images(image_to_verify,reference_image):
    img1 = cv2.imread(image_to_verify)
    img2 = cv2.imread(reference_image)
    plt.imshow(img1[:, :, ::-1])  # setting value as -1 to maintain saturation
    plt.show()
    plt.imshow(img2[:, :, ::-1])
    plt.show()