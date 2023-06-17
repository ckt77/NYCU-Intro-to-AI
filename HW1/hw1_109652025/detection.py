import cv2
import matplotlib.pyplot as plt
import numpy as np


def crop(x1, y1, x2, y2, x3, y3, x4, y4, img):
    """
    This function ouput the specified area (parking space image) of the input frame according to the input of four xy coordinates.

      Parameters:
        (x1, y1, x2, y2, x3, y3, x4, y4, frame)z

        (x1, y1) is the lower left corner of the specified area
        (x2, y2) is the lower right corner of the specified area
        (x3, y3) is the upper left corner of the specified area
        (x4, y4) is the upper right corner of the specified area
        frame is the frame you want to get it's parking space image

      Returns:
        parking_space_image (image size = 360 x 160)

      Usage:
        parking_space_image = crop(x1, y1, x2, y2, x3, y3, x4, y4, img)
    """
    left_front = (x1, y1)
    right_front = (x2, y2)
    left_bottom = (x3, y3)
    right_bottom = (x4, y4)
    src_pts = np.array([left_front, right_front, left_bottom,
                       right_bottom]).astype(np.float32)
    dst_pts = np.array([[0, 0], [0, 160], [360, 0], [
                       360, 160]]).astype(np.float32)
    projective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    croped = cv2.warpPerspective(img, projective_matrix, (360, 160))

    return croped


def detect(data_path, clf):
    """
    Please read detectData.txt to understand the format.
    Use cv2.VideoCapture() to load the video.gif.
    Use crop() to crop each frame (frame size = 1280 x 800) of video to get parking space images. (image size = 360 x 160)
    Convert each parking space image into 36 x 16 and grayscale.
    Use clf.classify() function to detect car, If the result is True, draw the green box on the image like the example provided on the spec.
    Then, you have to show the first frame with the bounding boxes in your report.
    Save the predictions as .txt file (ML_Models_pred.txt), the format is the same as Sample.txt.

      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    coordinates = []

    with open('data/detect/detectData.txt', 'r', encoding="utf-8") as file:
        for line in file.readlines():
            coordinates.append(line.strip().split(' '))
    coordinates = coordinates[1:]

    cap = cv2.VideoCapture('data/detect/video.gif')
    ret, frame = cap.read()
    predict_results = []
    frame_list = []

    while ret:
        predict_result = []
        for coordinate in coordinates:
            x1, y1, x2, y2, x3, y3, x4, y4 = coordinate[0], coordinate[1], coordinate[2], coordinate[
                3], coordinate[4], coordinate[5], coordinate[6], coordinate[7]
            img = crop(x1, y1, x2, y2, x3, y3, x4, y4, frame)
            img = cv2.resize(img, (36, 16))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            label = str(clf.classify([img.reshape(-1)]))
            predict_result.append(label + ' ')

        for i, coordinate in enumerate(coordinates):
            if predict_result[i] == '1 ':
                x1, y1, x2, y2, x3, y3, x4, y4 = int(coordinate[0]), int(coordinate[1]), int(coordinate[2]), int(coordinate[
                    3]), int(coordinate[4]), int(coordinate[5]), int(coordinate[6]), int(coordinate[7])
                pts = np.array([[x1, y1], [x2, y2], [x4, y4],
                                [x3, y3], [x1, y1]], np.int32)
                frame = cv2.polylines(
                    frame, [pts], False, (0, 255, 0), thickness=2)

        predict_result.append('\n')
        predict_results.append(predict_result)
        frame_list.append(frame)
        ret, frame = cap.read()

    cv2.imwrite('hw1_109652025_1.png', frame_list[0])
    cv2.destroyAllWindows()

    with open('ML_Models_pred.txt', 'w') as file:
        for predict_result in predict_results:
            file.writelines(predict_result)
    # End your code (Part 4)
