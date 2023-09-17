import os
import cv2

path = "E:\\SatdatRapi\V2\Dataset_ - Copy\\T"

for file in os.listdir(path):
    img = cv2.imread(path + "\\" + file)
    img = cv2.resize(img, (28, 28))
    cv2.imwrite(path + "\\" + file, img)
    print(file)
