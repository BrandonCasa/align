from math import floor, ceil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

fig, axs = plt.subplots(2, 3, sharey=False, tight_layout=True)

img = mpimg.imread('./images/im1_cropped.jpg')
axs[0,0].imshow(img)

y, x, _ = axs[0,1].hist(img[:, :, 0].ravel(), bins=256, fc='k', ec='k')
histogram = cv2.calcHist([img],[0],None,[256],[0,256])
axs[0,2].plot(histogram,color = 'r')

peak = x[np.where(y == y.max())]
print("Min: " + str(peak) + " - 22")
print("Max: 255")
min = floor(peak)-22
max = 255
mask = cv2.inRange(img, (min, 0, 0), (max, 255,255))
maskImg = img.copy()
maskImg[mask != 0] = [0,0,0]
maskImg[mask == 0] = [255,255,255]
axs[1,0].imshow(mask)

imGray = cv2.cvtColor(maskImg, cv2.COLOR_RGB2GRAY)
kernel = np.ones((3,3), np.uint8)
img_dilation = cv2.dilate(imGray, kernel, iterations=3)
img_erosion = cv2.erode(img_dilation, kernel, iterations=1)
cnts, hierarchy = cv2.findContours(img_erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
cv2.drawContours(img, cnts[:floor(len(cnts)*0.50)], -1, (0,255,0), thickness=-1)
axs[1,1].imshow(img)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite("./im1_mask.jpg", img)

plt.xlim([0,256])
plt.show()