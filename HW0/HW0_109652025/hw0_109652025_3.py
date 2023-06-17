import cv2
import numpy as np


img = cv2.imread("image.png")
height, width = img.shape[:2]

# translation
M_1 = np.float32([[1, 0, 200], [0, 1, 200]])
translated_img = cv2.warpAffine(img, M_1, (width, height))
cv2.imwrite('hw0_109652025_3_translation.png', translated_img)
cv2.destroyAllWindows()

# rotation
M_2 = cv2.getRotationMatrix2D((width/2, height/2), 270, 1)
rotated_img = cv2.warpAffine(img, M_2, (width, height))
cv2.imwrite('hw0_109652025_3_rotation.png', rotated_img)
cv2.destroyAllWindows()

# flipping
flipped_img = cv2.flip(img, 1)
cv2.imwrite('hw0_109652025_3_flipping.png', flipped_img)
cv2.destroyAllWindows()

# scaling
resized_img = cv2.resize(img, (width * 2, height * 3))
cv2.imwrite('hw0_109652025_3_scaling.png', resized_img)
cv2.destroyAllWindows()

# cropping
x = int(width/4)
y = int(height/4)
crop_width = int(width/2)
crop_height = int(height/2)
cropped_img = img[y:y+crop_height, x:x+crop_width]
cv2.imwrite('hw0_109652025_3_cropping.png', cropped_img)
cv2.destroyAllWindows()
