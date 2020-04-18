import cv2 as cv
import numpy as np
from FastUtils import *

def findText(image):
	image = image.astype('float32') / 255

	newHeight = 500
	newWidth = round(image.shape[1] * newHeight / image.shape[0])
	totalArea = newHeight * newWidth
	image = cv.resize(image, (newWidth, newHeight), interpolation = cv.INTER_AREA)
	gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # ---- Convert to grayscale
	blurred = convolve(gray, createGaussianKernel(1)) # ---- Gaussian Blur
	grad = gradient(blurred) # ---- Compute Gradient
	grad = grad - (np.max(grad)/6)
	# grad = grad - 0.5
	dilation = cv.dilate(grad, np.ones((5, 5)), iterations=1)
	dilation = dilation.astype(np.uint8) * 255
	_, contours, _ = cv.findContours(dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	im2 = image.copy()

	for contour in contours:
		area = cv.contourArea(contour)
		perim = cv.arcLength(contour, True)
		x, y, w, h = cv.boundingRect(contour)
		
		if (area > 0.0015 * totalArea):
			cv.rectangle(im2, (x, y), (x + w, y + h), (0, 0, 255), 2)
			
	#cv.imshow('Gradient', grad)
	#cv.imshow('Dilated', dilation)
	#cv.imshow('Detected Text', im2)
	return im2