import cv2 as cv
import numpy as np

#takes an image of board and isolates all the cards
img = cv.imread("setTest1.jpg", cv.IMREAD_GRAYSCALE)
if img is None:
    print("cannot read image")
    exit()

blur = cv.GaussianBlur(img, (11,11), 0)

canny = cv.Canny(blur, 100, 200) #uses canny edge detection to find contour

contours, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

cards = []
for i, cnt in enumerate(contours):
    #simplify contours by approximating them
    epsilon = 0.02*cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, epsilon, True)

    #if there are four points, it is a quadrilateral and is worth checking
    if len(approx) == 4:
        x, y, w, h = cv.boundingRect(approx)
        aspectRatio = float(w)/h

        #checking to make sure the contour is within the standard aspect ratio of a card
        if aspectRatio <= 2 and aspectRatio >= 1:
            #if it is, means that it is a card
            #only taking contours without parents, because it would contour both the inside and the outside of the card
            if hierarchy[0][i][3] == -1:
                cards.append(approx)

print(cards)
print("there are " + str(len(cards)) + " cards in play right now")


#create the regions of interest for each card
#create a bounding rectangle for each card, and use that to create regions of interest for cards
ROIs = []
img = cv.imread("setTest1.jpg")
for card in cards:
    x, y, w, h = cv.boundingRect(card)
    roi = img[y:y+h, x:x+w]
    ROIs.append(roi)

for i, roi in enumerate(ROIs):
    cv.imshow("Card " + str(i) + ":", roi)

cv.waitKey(0)
cv.destroyAllWindows()

