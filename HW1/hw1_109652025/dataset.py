import os
import cv2


def load_images(data_path):
    """
    Load all Images in the folder and transfer a list of tuples.
    The first element is the numpy array of shape (m, n) representing the image.
    (remember to resize and convert the parking space images to 36 x 16 grayscale images.)
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    # Begin your code (Part 1)
    dataset = []
    car_path = data_path + '/car'
    non_car_path = data_path + '/non-car'

    for file_name in os.listdir(car_path):
        img_path = os.path.join(car_path, file_name)
        img = cv2.imread(img_path)

        img = cv2.resize(img, (36, 16))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        dataset.append((img, 1))

    for file_name in os.listdir(non_car_path):
        img_path = os.path.join(non_car_path, file_name)
        img = cv2.imread(img_path)

        img = cv2.resize(img, (36, 16))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        dataset.append((img, 0))
    # End your code (Part 1)
    return dataset
