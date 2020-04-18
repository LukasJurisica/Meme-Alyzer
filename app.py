import cv2 as cv
import numpy as np
import os
from FastUtils import *

testing_path = "./testings/"
files = os.listdir(testing_path)

from FastUtils import *
from TemplateAnalysis import findTemplate
from TextAnalysis import findText

def analyzeMeme(image):
	score, name, match = findTemplate(image)
	print("This appears to be the", name, "template! with a score of", score)
	image = findText(image)
	image = scaleImageToHeight(image, 500)
	match = scaleImageToHeight(match, 500).astype('float32') / 255
	cv.imshow('Result', np.hstack((image, match)))
	#cv.imshow('Original', image)
	#cv.imshow('Template', match)

if __name__ == "__main__":
	'''
	args = sys.argv[1:]
	if (len(args) == 0):
		print("Please specify some image files to analyze (As CMD Line Arguments).")
	else:
		
		for filename in args:
			image = cv.imread(filename)
	
			if (image is not None):
				#image = image.astype('float32') / 255
				
				print("Analyzing:", filename)
				analyzeMeme(image)
				print()
			else:
				print("Invalid filename provided: " + filename + ". Image could not be loaded.")

		cv.waitKey(0)
		cv.destroyAllWindows()
	'''
	for file in files:
		image = cv.imread(testing_path + file)
		analyzeMeme(image)
		cv.waitKey(0)
		cv.destroyAllWindows()