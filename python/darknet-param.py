#!/usr/bin/env python
import os

from ctypes import *
import math
import random
import cv2
import time
import numpy as np
#import imutils

import rospy
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

DEBUG_VIEW = False

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

lib = CDLL("/home/nimbus/inspection_ws/src/my_image_detector/libdarknet.so", RTLD_GLOBAL)
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

def nparray_to_image(img):
    data = img.ctypes.data_as(POINTER(c_ubyte))
    image = ndarray_image(data, img.ctypes.shape, img.ctypes.strides)
    return image

def detect_video(net, meta, cam_size, zoom_size, crop_frame=False, cam=0, thresh=.5, hier_thresh=.5, nms=.45):
    global frame
    send_coord = rospy.Publisher('/bboxCoord', Float32MultiArray, queue_size=10)
    send_img = rospy.Publisher('/webcam', Image, queue_size = 1)

    vc = cv2.VideoCapture(cam, cv2.CAP_V4L2)
    vc.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, cam_size[0])
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_size[1])

    bridge = CvBridge()
    
    if DEBUG_VIEW:
        cv2.namedWindow("detection")

    rate = rospy.Rate(5)

    obj_msg = Float32MultiArray() # [conf_score, x_center, y_center, box_height, box_width, image_height, image_width]
    obj_msg.layout.dim.append(MultiArrayDimension())
    obj_msg.layout.dim.append(MultiArrayDimension())
    obj_msg.layout.dim[0].label = "entries"
    obj_msg.layout.dim[1].label = "det-info"


    avg_fps = []
    while not rospy.is_shutdown():
        ret, frame = vc.read()
        if not ret:
            break

        cw = frame.shape[1]
        ch = frame.shape[0]
        zw = zoom_size[0]
        zh = zoom_size[1]

        if crop_frame:
            cframe = frame[int((ch-zh)/2):int((ch+zh)/2),int((cw-zw)/2):int((cw+zw)/2),:]
            im = nparray_to_image(cframe)
            cv2.rectangle(frame, (int((cw-zw)/2),int((ch-zh)/2)), (int((cw+zw)/2),int((ch+zh)/2)), (0,0,255), 2)
        else:
            im = nparray_to_image(frame)

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
        free_image(im)
        free_detections(dets, num)

        classid = 0
        obj_msg.data = []
        Bboxes = []

        if len(res) > 0 :
            for r in res:
                bconf = r[1]
                (cx,cy,bh,bw) = np.array(r[2]).astype(np.int16)
                if r[0] == "blue":
                    bcolor = (255,0,0) #bgr
                    classid = 1
                elif r[0] == "red":
                    bcolor = (0,0,255)
                    classid = 2        
                elif r[0] == "yellow":
                    bcolor = (0,255,255)
                    classid = 3
                elif r[0] == "purple":
                    bcolor = (255,0,180)
                    classid = 4
                else:
                    bcolor = (0,255,0)
                    classid = 5

                if crop_frame:
                    cv2.rectangle(frame, (int((cw-zw)/2 + cx-bw/2), int((ch-zh)/2 + cy-bh/2)), (int((cw-zw)/2 + cx+bw/2), int((ch-zh)/2 + cy+bh/2)), bcolor, 2)
                    # print([(int((cw-zw)/2),int((ch-zh)/2)), (int((cw+zw)/2),int((ch+zh)/2))], [(int(cx-bw/2), int(cy-bh/2)), (int(cx+bw/2), int(cy+bh/2))] , [(int((cw-zw)/2 + cx-bw/2), int((ch-zh)/2 + cy-bh/2)), (int((cw-zw)/2 + cx+bw/2), int((ch-zh)/2 + cy+bh/2))])
                    (offset_w, offset_h) = ( (cw-zw)/2 , (ch-zh)/2 )
                else:    
                    cv2.rectangle(frame, (int(cx-bw/2), int(cy-bh/2)), (int(cx+bw/2), int(cy+bh/2)), bcolor, 2)
                    (offset_w, offset_h) = (0,0)
                
                bboxdata = [ classid, bconf, cx, cy, bh, bw, frame.shape[0], frame.shape[1], offset_h, offset_w ]
                Bboxes.append(bboxdata)
            
            Bboxes = np.array(Bboxes).astype(np.float16).squeeze()
            obj_msg.data = Bboxes

            # obj_msg.data = [0]*len(res)   

            # for i in range(len(res)):
            #     obj_msg.data[i] = np.array(Bboxes[i])


        else:
            obj_msg.data = [0.0]*10

        send_coord.publish(obj_msg)

        img_msg = bridge.cv2_to_imgmsg(frame, "bgr8")
        send_img.publish(img_msg)

        rate.sleep()

        if DEBUG_VIEW:
            cv2.imshow("detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if rospy.is_shutdown():
            vc.release()


    rospy.spin()
    if DEBUG_VIEW:
        cv2.destroyWindow("detection")
    

if __name__ == "__main__":
    rospy.init_node('darknet')
    #target = "multiball"
    target = rospy.get_param("~target_obj")
    cam_size = [rospy.get_param("~cam_width"), rospy.get_param("~cam_height")]
    zoom_size = [rospy.get_param("~zoom_width"), rospy.get_param("~zoom_height")]
    crop_frame = rospy.get_param("~crop_frame")

    net = load_net(b"/home/nimbus/inspection_ws/src/my_image_detector/cfg/yolov3-tiny-obj-test.cfg", 
        b"/home/nimbus/inspection_ws/src/my_image_detector/backup/"+target+"/yolov3-tiny-obj-"+target+"_final.weights", 0)
    meta = load_meta(b"/home/nimbus/inspection_ws/src/my_image_detector/data/"+target+"/obj.data")

    try:
        detect_video(net, meta, cam_size, zoom_size, crop_frame, 0)
    except rospy.ROSInterruptException:
        pass
