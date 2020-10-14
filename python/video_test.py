#!/usr/bin/env python
import os

from ctypes import *
import math
import random
import cv2
import time

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
lib = CDLL("/home/robot/catkin_ws/src/darknet-yolo/libdarknet.so", RTLD_GLOBAL)
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
    t_start = time.time()
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
    t_end = time.time()

    print("inference speed {} FPS".format(1/(t_end - t_start)))

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

def detect_video(net, meta, cam=0, thresh=.5, hier_thresh=.5, nms=.45):
    cv2.namedWindow("yolo_demo")
    vc = cv2.VideoCapture(cam)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()

        while rval:
            rval, im = vc.read()
            cv2.imshow("yolo_demo", im)
            '''
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

            obj_location.confidence = res[0][1]
            obj_location.centerX = int(res[0][2][0])
            obj_location.centerY = int(res[0][2][1])
            obj_location.height = int(res[0][2][2])
            obj_location.width = int(res[0][2][3])

            send_coord.publish(obj_location)
            '''
    else:
        rval = False
        print('Reading failed')

if __name__ == "__main__":
    net = load_net(b"cfg/yolov3-tiny-obj.cfg", b"/home/robot/catkin_ws/src/darknet-yolo/backup/bolt/yolov3-tiny-obj_800.weights", 0)
    meta = load_meta(b"data/bolt/obj.data")
    r = detect(net, meta, b"data/bolt/img/IMG_4265.JPG")
    #process_coord(r)
    #publish(r)
    #detect_video(net, meta, 0)