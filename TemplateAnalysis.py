import cv2 as cv
import numpy as np
import os
from FastUtils import *

template_path = "./templates/"
files = os.listdir(template_path)
comparisonScale = 500

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
		
		score = 0
		#good_matches = []
		
		for i, (m, n) in enumerate(matches):
			score += 1
			if(m.distance < 0.7 * n.distance):
				score += 1
		#		good_matches.append([m])
				
		#img3 = cv.drawMatchesKnn(source,kpt1,target,kpt2,good_matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
		#cv.imshow('Temp', img3)
		#cv.waitKey(0)
		#cv.destroyAllWindows()

		scores.append(score)

	#print(scores)
	n = np.argmax(scores)
	return scores[n], files[n], cv.imread(template_path + files[n])
