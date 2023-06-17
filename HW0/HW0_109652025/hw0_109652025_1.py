import cv2


img = cv2.imread('image.png')
label_1 = cv2.rectangle(img, (608, 616), (721, 505), (0, 0, 255), 5)
label_2 = cv2.rectangle(img, (836, 557), (916, 477), (0, 0, 255), 5)
label_3 = cv2.rectangle(img, (1073, 726), (1180, 600), (0, 0, 255), 5)
label_4 = cv2.rectangle(img, (985, 468), (1042, 417), (0, 0, 255), 5)
label_5 = cv2.rectangle(img, (994, 383), (1042, 346), (0, 0, 255), 5)
label_6 = cv2.rectangle(img, (1042, 314), (1088, 282), (0, 0, 255), 5)

cv2.imwrite('hw0_109652025_1.png', img)
cv2.destroyAllWindows()
