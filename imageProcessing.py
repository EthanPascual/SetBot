import cv2 as cv
import numpy as np


def createContour(image):
    img = cv.imread(image, cv.IMREAD_GRAYSCALE)
    blurred_img = cv.GaussianBlur(img, (15, 15), 0)
    _, thresh = cv.threshold(blurred_img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    canny = cv.Canny(thresh, 50, 150)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv.morphologyEx(canny, cv.MORPH_OPEN, kernel)
    contours, hierarchy = cv.findContours(opening, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours[-1]

diamondContour = createContour("daimond.jpg")
squiggleContour = createContour("squiggle.jpg")
ovalContour = createContour("oval.jpg")

def findShape(contour):
    match = float("inf")
    shape = None
    ret = cv.matchShapes(diamondContour, contour, 1, 0)
    if ret < match:
        shape = "Diamond"
        match = ret
    ret = cv.matchShapes(ovalContour, contour, 1, 0)
    if ret < match:
        shape = "Oval"
        match = ret
    ret = cv.matchShapes(squiggleContour, contour, 1, 0)
    if ret < match:
        shape = "Squiggle"
        match = ret

    return shape

#takes an image of board and isolates all the cards
img = cv.imread("setTest2.jpg", cv.IMREAD_GRAYSCALE)
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

print("there are " + str(len(cards)) + " cards in play right now")


#create the regions of interest for each card
#create a bounding rectangle for each card, and use that to create regions of interest for cards
ROIs = []
img = cv.imread("setTest2.jpg")
for card in cards:
    x, y, w, h = cv.boundingRect(card)
    roi = img[y:y+h, x:x+w]
    ROIs.append(roi)

for i, roi in enumerate(ROIs):
    roi_gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(roi_gray, (15, 15), 0)
    thresh = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
    canny = cv.Canny(thresh, 100, 200)

    contours, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    count = 0
    minArea = 1000
    shapeCheck = []
    for x, contour in enumerate(contours): #counting how many shapes

        #checks to see if contour is closed, it has no children, and the area greater than a certain value (sometimes noise shows up)
        if cv.contourArea(contour) > cv.arcLength(contour, True) and hierarchy[0][x][3] == -1 and cv.contourArea(contour) > minArea:
            count += 1
            if len(shapeCheck) == 0:
                shapeCheck = contour
    shape = findShape(shapeCheck)

    cv.imshow("Card " + str(i) + ": " + str(count) + " " + shape + " shapes", roi)




#determine what each card is
cv.waitKey(0)
cv.destroyAllWindows()

