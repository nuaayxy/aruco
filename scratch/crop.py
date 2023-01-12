import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


import cv2
from sqlalchemy import true
name = '301.png'
image = cv2.imread(name)
original = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
# print(thresh)
# # Morph close
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
# #close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

# # Find contours and filter for QR code
# cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print(len(cnts))
# print(cnts)
# cnts1 = cnts[1].copy()
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]

# x0 = 0
# y0 = 0
# for c in cnts:
#     peri = cv2.arcLength(c, True)
#     approx = cv2.approxPolyDP(c, 0.04 * peri, True)
#     x,y,w,h = cv2.boundingRect(approx)
#     area = cv2.contourArea(c)
#     ar = w / float(h)
#     if len(approx) == 4 and area > 1000 and (ar > .85 and ar < 1.3):
#         cv2.rectangle(image, (x, y-1), (x + w, y + h), (36,255,12), 3)
#         ROI = original[y:y+h-1, x:x+w]
#         x0 = x
#         y0 = y
#         #cv2.imwrite('ROI.png', ROI)

# x1 = 0
# y1 = 0
# for c in cnts1:
#     peri = cv2.arcLength(c, True)
#     approx = cv2.approxPolyDP(c, 0.04 * peri, True)
#     x,y,w,h = cv2.boundingRect(approx)
#     area = cv2.contourArea(c)
#     ar = w / float(h)
#     if len(approx) == 4 and area > 1000 and (ar > .85 and ar < 1.3):
#         cv2.rectangle(image, (x, y-1), (x + w, y + h), (36,255,12), 3)
#         ROI = original[y:y+h-1, x:x+w]
#         x1 =  x + w
#         y1 =  y + h -1
#         #cv2.imwrite('ROI.png', ROI)

# ROI = original[y0:y1, x0:x1]

height = image.shape[0]
width = image.shape[1]

a = [0,0]
b =[0, 0]
flag = False
for i in range(height):
    for j in range(width):
        if gray[i][j] < 50:
            a = [i, j]
            flag = true
            print(a)
            break
    if flag == true:
        break

flag = True
for i in reversed(range(height)):
    for j in reversed(range(width)):
        if gray[i][j] < 50:
            b = [i, j]
            flag = true
            print(b)
            break
    if flag == true:
        break

ROI = original[a[0]:b[0], a[1]:b[1]]
cv2.imwrite(name, ROI)
# # cv2.imshow('thresh', thresh)
# cv2.imshow('image', image)
cv2.imshow('ROI', ROI)
cv2.waitKey()     