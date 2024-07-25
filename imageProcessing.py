import cv2 as cv
import numpy as np

img = cv.imread("setTest2.jpg", cv.IMREAD_GRAYSCALE)
if img is None:
    print("cannot read image")
    exit()

blur = cv.GaussianBlur(img, (11,11), 0)

canny = cv.Canny(blur, 100, 200)

contours, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

cards = []
for i, cnt in enumerate(contours):
    epsilon = 0.02*cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, epsilon, True)
    if len(approx) == 4:
        x, y, w, h = cv.boundingRect(approx)
        aspectRatio = float(w)/h
        if aspectRatio <= 2 and aspectRatio >= 1:
            if hierarchy[0][i][3] == -1:
                cards.append(approx)

print(cards)
print("there are " + str(len(cards)) + " cards in play right now")
img = cv.imread("setTest2.jpg")
cv.drawContours(img, cards, -1, (0, 255, 0), 5)


cv.imshow("Test", canny)
cv.imshow("contours", img)
cv.waitKey(0)
cv.destroyAllWindows()