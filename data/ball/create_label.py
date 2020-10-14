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
import glob

# input
target = "ball"

# global variables

# Load annotations
label_dir = os.path.abspath("./data/"+target)
dataset_dir = os.path.abspath("./data/"+target+"/img/")
annotations = json.load(open(os.path.join(label_dir, "annotation.json")))

# Skip unannotated images.
annotations = [a for a in annotations if type(a['Label'])==dict]

def add_label(class_name, class_idx, dic, im):
	#global turn

	turn = False
	save = True
	if im.shape[0] > im.shape[1]: # height > width --> width > height
		turn = True
		im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)

	if class_name in dic.keys():
		for r in dic.get(class_name):
			a1, a2, a3, a4 = bounding_box(im, r['geometry'], turn)
			f.write("{} {:2.6f} {:2.6f} {:2.6f} {:2.6f}\n".format(class_idx, a1, a2, a3, a4))
		return save, turn#, gt
	else:
		save = False
		return save, turn#, gt

def bounding_box(image, polygon, turn):
	height = image.shape[0]
	width = image.shape[1]

	#print("{}x{}".format(width, height))

	points = []
	for i in range(len(polygon)):
		points.append([polygon[i]['x'], polygon[i]['y']])

		x_coordinates, y_coordinates = zip(*points)

		x_min = min(x_coordinates)
		x_max = max(x_coordinates)
		y_min = min(y_coordinates)
		y_max = max(y_coordinates)
	'''
	if not turn: # original
		_x_min = width-y_max
		_x_max = width-y_min
		_y_min = x_min
		_y_max = x_max

		x_cen = (_x_min+_x_max)/(2*width)
		y_cen = (_y_min+_y_max)/(2*height)
		r_width = abs(_x_max-_x_min)/width
		r_height = abs(_y_max-_y_min)/height

		gt = cv2.rectangle(image, (_x_min, _y_min), (_x_max, _y_max), color=(0,0,255), thickness = 3)
	else: # rot 90 ccw
	'''
	x_cen = (x_min+x_max)/(2*width)
	y_cen = (y_min+y_max)/(2*height)
	r_width = abs(x_max-x_min)/width
	r_height = abs(y_max-y_min)/height

	#gt = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0,0,255), thickness = 3)

	return x_cen, y_cen, r_width, r_height#, gt

# generate dataset as Yolomark-format
f1 = open("./data/"+target+"/train.txt", "w")
print("-------------------------------------------------------------")
print("* Annotation: {} \n* Dataset: {}".format(os.path.join(label_dir, "annotation.json"), os.path.abspath("./data/"+target+"/img/")))
for a in annotations:
	print("   processing --> {}".format(a['External ID']))

	if a['External ID'] != "IMG_2858.JPG":
		dic = a['Label']
		path = os.path.join(dataset_dir, a['External ID'])
		img = cv2.imread(path, cv2.IMREAD_COLOR)
		f = open(dataset_dir+'/'+a['External ID'][:-4]+".txt", "w") #save txt to /img/img/

		if target == "bolt":
			obj = 'Bolts'
		#elif target == "rivet":
			#obj = 'Rivets'
		elif target == "ball":
			obj = 'ball'
		save, turn = add_label(obj, 0, dic, img)
		if turn:
			img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
		if save:
			f1.write("data/"+target+"/img/{}\n".format(a['External ID'])) # for train.txt
			cv2.imwrite(dataset_dir+'/'+a['External ID'], img) #save image to /img/img
			#cv2.imwrite(dataset_dir+"/gt/"+a['External ID'], gt)

	f.close()
f1.close()

# remove empty txt files
print("- empty files removed")
for files in glob.glob(dataset_dir+"/img/*.txt"):
	if os.stat(files).st_size == 0:
		os.remove(files)
print("-------------------------------------------------------------")