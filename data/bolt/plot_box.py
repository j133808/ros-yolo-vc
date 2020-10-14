# python plot grount-truth bbox
# BOLT

import os
import json

import numpy as np
import pandas as pd
import skimage.draw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as img
import cv2
import math
import glob

from matplotlib.collections import PatchCollection

# Load annotations
label_dir = os.path.abspath("./")
dataset_dir = os.path.abspath("./img/")
#results = json.load(open(os.path.join(label_dir, "result.json")))

#width = 3024
#height = 4032

def plot(image, label):
	width = image.shape[0]
	height = image.shape[1]

	for i in label:
		coord = i.split()

		x_cen = float(coord[1])*width #(x_min+x_max)/(2*width)
		y_cen = float(coord[2])*height #(y_min+y_max)/(2*height)
		r_width = float(coord[3])*width #abs(x_max-x_min)/width
		r_height = float(coord[4])*height #abs(y_max-y_min)/height

		x_min = int(x_cen-r_width/2)
		x_max = int(x_cen+r_width/2)
		y_min = int(y_cen-r_height/2)
		y_max = int(y_cen+r_height/2)

		#print("{}: ({},{}), ({},{})".format(coord, x_min, y_min, x_max, y_max))

		gt = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0,0,255), thickness = 3)

	return gt


# Add images
print("Start plotting...")

for infile in sorted(glob.glob(os.path.join(dataset_dir, '*.JPG'))):
	if infile.find("IMG_6034.JPG") != -1:
		print("processing --> {}".format(infile[-12:]))
		image = cv2.imread(infile, cv2.IMREAD_COLOR)
		label = open(infile[:-4]+'.txt', "r")
		gt = plot(image, label)
		cv2.imwrite(infile[:-4]+'_gt_check.JPG', gt)
		label.close()

print("done.")