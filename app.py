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

if __name__ == "__main__":
	for file in files:
		image = cv.imread(testing_path + file)
		analyzeMeme(image)
		cv.waitKey(0)
		cv.destroyAllWindows()