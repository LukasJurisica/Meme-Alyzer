import cv2 as cv
import numpy as np
import os
from FastUtils import *

template_path = "./templates/"
files = os.listdir(template_path)
comparisonScale = 300 # ---- CHANGE THIS TO 500 FOR PRODUCTION, MAKES IT SLOWER BUT MORE ACCURATE

def findTemplate(source):
	source = cv.cvtColor(source, cv.COLOR_BGR2GRAY)
	source = scaleImageToWidth(source, comparisonScale)
	
	scores = []
	for file in files:
		target = cv.imread(template_path + file, cv.IMREAD_GRAYSCALE) # ---- load template image
		target = scaleImageToWidth(target, comparisonScale) # ---- Convert to 500x_ image (same aspect ratio)

		#create an ORB detector, find descriptors and keypoints
		orb = cv.ORB_create(nfeatures = 10000)	
		kpt1, desc1 = orb.detectAndCompute(source, None)
		kpt2, desc2 = orb.detectAndCompute(target, None)

		#Transform them to float np-arrays
		kps1 = np.float32([kpnts.pt for kpnts in kpt1])
		kps2 = np.float32([kpnts.pt for kpnts in kpt2])
		
		#Create Brute Force Matcher
		bf = cv.BFMatcher()

		#Match with K- nearest neighbours
		matches = bf.knnMatch(desc1, desc2, k = 2)
		matchMask = np.full((len(matches), 2), 0)
		
		score = 0
		for i, (m, n) in enumerate(matches):
			if(m.distance < 0.75 * n.distance):
				score += 1

		scores.append(score)

	#print(scores)
	n = np.argmax(scores)
	return scores[n], files[n], cv.imread(template_path + files[n])