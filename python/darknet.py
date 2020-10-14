#!/usr/bin/env python
import os

from ctypes import *
import math
import random
import cv2
import time
import numpy as np

import rospy
from std_msgs.msg import Float32MultiArray
from yolo.msg import yoloBBox

send_coord = rospy.Publisher('/bboxCoord', yoloBBox,queue_size=10)

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
#lib = CDLL("libdarknet.so", RTLD_GLOBAL)
lib = CDLL("/home/nimbus/catkin_ws/src/darknet-yolo/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

ndarray_image = lib.ndarray_to_image
ndarray_image.argtypes = [POINTER(c_ubyte), POINTER(c_long), POINTER(c_long)]
ndarray_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res

def publish(data):
    rate = rospy.Rate(10)

    obj_location = yoloBBox()
    while not rospy.is_shutdown():
        obj_location.confidence = data[0][1]
        obj_location.centerX = int(data[0][2][0])
        obj_location.centerY = int(data[0][2][1])
        obj_location.height = int(data[0][2][2])
        obj_location.width = int(data[0][2][3])

        #print("conf: {} / [{},{},{},{}]".format(obj_location.confidence, obj_location.centerX, obj_location.centerY, obj_location.height, obj_location.width))

        send_coord.publish(obj_location)

        rate.sleep()
    rospy.spin()

def process_coord(rs):
    #crd = []
    #for r in rs:
    print("conf: {} / [{},{},{},{}]".format(rs[0][1], rs[0][2][0], rs[0][2][1], rs[0][2][2], rs[0][2][3]))

def nparray_to_image(img):

    data = img.ctypes.data_as(POINTER(c_ubyte))
    image = ndarray_image(data, img.ctypes.shape, img.ctypes.strides)

    return image

def draw_grid(img, idx):
    h = img.shape[0]
    w = img.shape[1]
    colors = (255,255,255)
    
    cv2.line(img, (int(w/3), 0),(int(w/3), h), colors, 2, 1)
    cv2.line(img, (int(w/3*2), 0),(int(w/3*2), h), colors, 2, 1)
    cv2.line(img, (0, int(h/3)),(w, int(h/3)), colors, 2, 1)
    cv2.line(img, (0, int(h/3*2)),(w, int(h/3*2)), colors, 2, 1)
    
    windowsize_r = int(img.shape[0]/3)-1
    windowsize_c = int(img.shape[1]/3)-1

    # for tracking
    imgs = []
    rects = []
    for r in range(0,img.shape[0] - windowsize_r, windowsize_r):
        for c in range(0,img.shape[1] - windowsize_c, windowsize_c):
            window = img[r:r+windowsize_r,c:c+windowsize_c]
            imgs.append(window)
            rect = np.ones(window.shape, dtype=np.uint8) * 255
            rect = cv2.addWeighted(window, 0.5, rect, 0.5, 1.0)
            rects.append([rect, r, c])
    '''        
    i = idx
    
    if i is not 4:
        r = rects[i][1]
        c = rects[i][2]
        img[r:r+windowsize_r, c:c+windowsize_c] = rects[i][0]
    else:        
        cv2.line(img, (int(w/9*4), int(h/3)),(int(w/9*4), int(h/3*2)), colors, 1, 1)
        cv2.line(img, (int(w/9*5), int(h/3)),(int(w/9*5), int(h/3*2)), colors, 1, 1)
        cv2.line(img, (int(w/3), int(h/9*4)),(int(w/3*2), int(h/9*4)), colors, 1, 1)
        cv2.line(img, (int(w/3), int(h/9*5)),(int(w/3*2), int(h/9*5)), colors, 1, 1)
        
        windowsize_r_l = int(img.shape[0]/9)
        windowsize_c_l = int(img.shape[1]/9)

        # for localization
        imgs_l = []
        rects_l = []
        for r in range(windowsize_r, windowsize_r*2, windowsize_r_l):
            for c in range(windowsize_c, windowsize_c*2, windowsize_c_l):
                window = img[r:r+windowsize_r_l,c:c+windowsize_c_l]
                imgs_l.append(window)
                rect = np.ones(window.shape, dtype=np.uint8) * 255
                rect = cv2.addWeighted(window, 0.5, rect, 0.5, 1.0)
                rects_l.append([rect, r, c])
                
        i = np.random.randint(low=0, high=len(imgs_l)-1)
        r = rects_l[i][1]
        c = rects_l[i][2]
        img[r:r+windowsize_r_l, c:c+windowsize_c_l] = rects_l[i][0]
    '''
    r = rects[i][1]
    c = rects[i][2]
    img[r:r+windowsize_r, c:c+windowsize_c] = rects[i][0]

def draw_small_grid(img,center):
        cv2.line(img, (int(w/9*4), int(h/3)),(int(w/9*4), int(h/3*2)), colors, 1, 1)
        cv2.line(img, (int(w/9*5), int(h/3)),(int(w/9*5), int(h/3*2)), colors, 1, 1)
        cv2.line(img, (int(w/3), int(h/9*4)),(int(w/3*2), int(h/9*4)), colors, 1, 1)
        cv2.line(img, (int(w/3), int(h/9*5)),(int(w/3*2), int(h/9*5)), colors, 1, 1)
        
        windowsize_r_l = int(img.shape[0]/9)
        windowsize_c_l = int(img.shape[1]/9)

        # for localization
        imgs_l = []
        rects_l = []
        idx=0
        for r in range(windowsize_r, windowsize_r*2, windowsize_r_l):
            for c in range(windowsize_c, windowsize_c*2, windowsize_c_l):
                window = img[r:r+windowsize_r_l,c:c+windowsize_c_l]
                imgs_l.append(window)
                rect = np.ones(window.shape, dtype=np.uint8) * 255
                rect = cv2.addWeighted(window, 0.5, rect, 0.5, 1.0)
                rects_l.append([rect, r, c])
                if center[1]>=r and center[1]<=r+windowsize_r_l and center[0]>=c and center[0]<=c+windowsize_c_l:
                    i = idx
                
        #i = np.random.randint(low=0, high=len(imgs_l)-1)
        r = rects_l[i][1]
        c = rects_l[i][2]
        img[r:r+windowsize_r_l, c:c+windowsize_c_l] = rects_l[i][0]

def detect_video(net, meta, cam=0, thresh=.5, hier_thresh=.5, nms=.45):
    vc = cv2.VideoCapture(cam)
    cv2.namedWindow("detection")
    cv2.namedWindow("tracking")

    rate = rospy.Rate(10)

    obj_location = yoloBBox()

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()

        avg_fps = []
        img = cv2.imread('/home/nimbus/catkin_ws/src/darknet-yolo/images/IMG_6141.JPG')
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
        img=s
        vf = cv2.xfeatures2d.SURF_create()
        prior_keypoints, prior_descriptors = vf.detectAndCompute(img, None)


        while not rospy.is_shutdown():
            rval, frame = vc.read()
            flag = False
            im = nparray_to_image(frame)
            grid_im = frame
            
            # tracking process
            if flag == False:
		        # Convert BGR to HSV and parse HSV
				hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
				h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
				img=h
				#split image into segments
				imgs = []
				windowsize_r = int(img.shape[0]/3)-1
				windowsize_c = int(img.shape[1]/3)-1
				
				for r in range(0,img.shape[0] - windowsize_r, windowsize_r):
					for c in range(0,img.shape[1] - windowsize_c, windowsize_c):
						window = img[r:r+windowsize_r,c:c+windowsize_c]
						imgs.append(window)
						
				# match from small patch
				#measure matching rate for each segments
				mat_rate = []
				for i in imgs:
					keypoints, descriptors = vf.detectAndCompute(i, None)
					bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
					matches = bf.match(descriptors,prior_descriptors)
					if not matches:
						match_rate = 0
					else:
						match_rate = len(matches)
					mat_rate.append(match_rate)	
				
				print((mat_rate, sum(mat_rate)))
				
				if sum(mat_rate) > 55:
				    flag = True
				    
				direction = np.argwhere(mat_rate == np.amax(mat_rate))
				direction = direction.flatten()
				
				if len(direction) > 1:
					#print("pick random move")
					direction = direction[np.random.randint(low=0, high=len(direction)-1)]
				else:
					direction = direction[0]
					
				draw_grid(grid_im,direction)

            else:
		        # Inference process
		        t_start = time.time()
		        num = c_int(0)
		        pnum = pointer(num)
		        predict_image(net, im)
		        dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
		        num = pnum[0]
		        if (nms): do_nms_obj(dets, num, meta.classes, nms);

		        res = []
		        for j in range(num):
		            for i in range(meta.classes):
		                if dets[j].prob[i] > 0:
		                    b = dets[j].bbox
		                    res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
		        res = sorted(res, key=lambda x: -x[1])
		        free_image(im)
		        free_detections(dets, num)
		        t_end = time.time()

		        fps = 1/(t_end-t_start)
		        print("{}x{}: {} object found / {:4.4f} FPS".format(frame.shape[0], frame.shape[1] , len(res), fps ))
		        avg_fps.append(fps)

		        if len(res) > 0 :
		            cx = int(res[0][2][0])
		            cy = int(res[0][2][1])
		            bconf = res[0][1]
		            bh = int(res[0][2][2])
		            bw = int(res[0][2][3])

		            cv2.rectangle(frame, (int(cx-bw/2), int(cy-bh/2)), (int(cx+bw/2), int(cy+bh/2)), (0, 0, 255), 2)
                draw_small_grid(grid_im, [cx,cy])

		        cv2.imshow("detection", frame)
		        cv2.imshow("tracking", grid_im)
		        cv2.waitKey(5)

		        if len(res)>0 :
		            obj_location.confidence = bconf
		            obj_location.centerX = cx
		            obj_location.centerY = cy
		            obj_location.height = bh
		            obj_location.width = bw
		        else:
		            obj_location.confidence = 0
		            obj_location.centerX = 1
		            obj_location.centerY = 0
		            obj_location.height = 1
		            obj_location.width = 0
		        send_coord.publish(obj_location)
		        
            rate.sleep()
        rospy.spin()
        cv2.destroyWindow("detection")
        cv2.destroyWindow("tracking")
        print("Average FPS: {}".format(sum(avg_fps)/len(avg_fps)))

    else:
        rval = False
        print('Reading failed')

if __name__ == "__main__":
    rospy.init_node('darknet')
    net = load_net(b"cfg/yolov3-tiny-obj-test.cfg", b"/home/nimbus/catkin_ws/src/darknet-yolo/backup/pokeball/yolov3-tiny-obj_final.weights", 0)
    meta = load_meta(b"data/pokeball/obj.data")
    #r = detect(net, meta, b"data/bolt/img/IMG_4265.JPG")
    #process_coord(r)
    #publish(r)
    detect_video(net, meta, 0)
