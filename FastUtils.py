import math
import numpy as np
import cv2 as cv
import sys
from scipy import signal
	
def flipKernel(kernel):
	return np.flip(kernel, (0, 1))

def normalizeKernel(kernel):
	return kernel / np.sum(kernel)

def createGaussianKernel(sigma):
	size = (2 * math.ceil(3 * sigma)) + 1
	offset = math.floor(size / 2)
	s = 2 * sigma * sigma;
	result = np.zeros([size, size])
	
	for i in range(size):
		for j in range(size):
			r = pow(i - offset, 2) + pow(j - offset, 2)
			result[i][j] = math.exp(-r / s) / (math.pi * s)

	return normalizeKernel(result)

def convolve2d(image, kernel):
	return signal.convolve2d(image, kernel, boundary='symm', mode='same')

def convolve3d(image, kernel):
	image_height, image_width, n_channels = image.shape
	result = np.zeros_like(image)
	
	for c in range(n_channels):
		result[:image_height, :image_width, c] = convolve2d(image[:image_height, :image_width, c], kernel)

	return result

def convolve(image, kernel):
	if (len(image.shape) == 2):
		return convolve2d(image, kernel)
	elif (len(image.shape) == 3):
		return convolve3d(image, kernel)
	else:
		return image

def gradient(image):
	h_sobel = np.array([ [ 1, 0,-1], [ 2, 0,-2], [ 1, 0,-1] ])
	v_sobel = np.array([ [ 1, 2, 1], [ 0, 0, 0], [-1,-2,-1] ])
	h_gradient = convolve(image, h_sobel)
	v_gradient = convolve(image, v_sobel)
	#angles = np.arctan2(h_gradient, v_gradient)
	return np.sqrt(np.square(h_gradient) + np.square(v_gradient))
	
def zoreThresholding(image, count):
	count -= 1
	#result = np.zeros((image.shape[0], image.shape[1], count))
	image = image * count
	image = np.round(image)
	image = image / count
	return image

def zoreThresholding2(image, low, high):
	result = np.zeros((image.shape[0], image.shape[1], 3))
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):	
			if (image[i, j] <= low):
				result[i, j, 0] = 1
			elif (image[i, j] >= high):
				result[i, j, 2] = 1
			else:
				result[i, j, 1] = 1
				
	return result
	
def stackImages(images):
	result = images[0]
	for i in range(len(images) - 1):
		temp = images[i+1]
		if (len(temp.shape) == 2):
			temp = cv.cvtColor(temp, cv.COLOR_GRAY2BGR)
		result = np.hstack((result, temp))
	return result
	
def scaleImageToHeight(image, newHeight):
	newWidth = round(image.shape[1] * newHeight / image.shape[0])
	return cv.resize(image, (newWidth, newHeight), interpolation = cv.INTER_AREA)
	
def scaleImageToWidth(image, newWidth):
	newHeight = round(image.shape[0] * newWidth / image.shape[1])
	return cv.resize(image, (newWidth, newHeight), interpolation = cv.INTER_AREA)