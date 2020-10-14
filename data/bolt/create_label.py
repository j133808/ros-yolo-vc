# python convert labelbox coco json to txt file
# BOLT

import os
import json

import numpy as np
import pandas as pd
import skimage.draw
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2
import math

# Load annotations
label_dir = os.path.abspath("./")
dataset_dir = os.path.abspath("./img/")
annotations = json.load(open(os.path.join(label_dir, "annotation.json")))

#Skip unannotated images.
annotations = [a for a in annotations if type(a['Label'])==dict]

def add_label(class_name, class_idx, dic, im):
	if class_name in dic.keys():
		for r in dic.get(class_name):
			a1, a2, a3, a4, gt, im, turn = bounding_box(im, r['geometry'])
			f.write("{} {:2.6f} {:2.6f} {:2.6f} {:2.6f}\n".format(class_idx, a1, a2, a3, a4))
		return im, gt
	else:
		return im, im

def bounding_box(image, polygon):
	turn = False

	width = image.shape[0]
	height = image.shape[1]

	points = []
	for i in range(len(polygon)):
		points.append([polygon[i]['x'], polygon[i]['y']])

	x_coordinates, y_coordinates = zip(*points)

	if width > height:
		turn = True
		#temp = x_coordinates
		#x_coordinates = y_coordinates
		#y_coordinates = temp

	x_min = min(x_coordinates)
	x_max = max(x_coordinates)
	y_min = min(y_coordinates)
	y_max = max(y_coordinates)

	x_cen = (x_min+x_max)/(2*width)
	y_cen = (y_min+y_max)/(2*height)
	r_width = abs(x_max-x_min)/width
	r_height = abs(y_max-y_min)/height

	'''
	if turn:
		image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
		gt = np.rot90(cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0,0,255), thickness = 3), k=1)
	else:
		gt = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0,0,255), thickness = 3)
	'''

	gt = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0,0,255), thickness = 3)
	#if turn:
	#	image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

	return x_cen, y_cen, r_width, r_height, gt, image, turn

# Add images
print("Start convering labels...")

f1 = open("train.txt", "w")
for a in annotations:
	print("processing --> {}".format(a['External ID']))

	dic = a['Label']
	path = os.path.join(dataset_dir, a['External ID'])
	im = cv2.imread(path, cv2.IMREAD_COLOR)
	#f = open(path[:-4]+".txt", "w")
	f = open(dataset_dir+"/img/"+a['External ID'][:-4]+".txt", "w")

	f1.write("bolt/data/img/img/{}\n".format(a['External ID']))

	im, gt = add_label('Bolts', 0, dic, im)
	cv2.imwrite(dataset_dir+"/img/"+a['External ID'], im)
	cv2.imwrite(dataset_dir+"/gt/"+a['External ID'], gt)

	f.close()

f1.close()
print("done.")