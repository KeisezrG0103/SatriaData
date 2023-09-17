import cv2 as cv 

path = "E:\\SatdatRapi\\V2\\TESTEDITED\\DataTest93.png"

img = cv.imread(path)
img = cv.resize(img, (1000, 400))

# invert
img = cv.bitwise_not(img)



#save
cv.imwrite("E:\\SatdatRapi\\V2\\TESTEDITED\\DataTest93.png", img)