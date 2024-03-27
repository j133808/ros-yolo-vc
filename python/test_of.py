#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

fwidth = 400
fheight = 400

def ofvc(cam=0):
    pub_rgb = rospy.Publisher('/off_rgb', Image, queue_size = 1)
    pub_hsv = rospy.Publisher('/off_hsv', Image, queue_size = 1)
    rate = rospy.Rate(10)

    cap = cv2.VideoCapture(cam, cv2.CAP_V4L2)
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, fwidth)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, fheight)

    print(cap.isOpened())
    bridge = CvBridge()

    # # initial value
    # ret, frame1 = cap.read()
    # prv_img = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    # hsv = np.zeros_like(frame1)
    # hsv[..., 1] = 255

    start = rospy.get_rostime()

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            print("vc read error")
            break

        now = rospy.get_rostime()
        if now.secs - start.secs < 1:
            prv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = np.zeros_like(frame)
            hsv[..., 1] = 255
            print("waiting for input...")
        else:
            nxt_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prv_img, nxt_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang*180/np.pi/2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            msg = bridge.cv2_to_imgmsg(frame, "rgb8")
            pub_rgb.publish(msg)

            msg = bridge.cv2_to_imgmsg(rgb, "rgb8")
            pub_hsv.publish(msg)

            prv_img = nxt_img

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if rospy.is_shutdown():
            cap.release()

if __name__ == '__main__':
    rospy.init_node('image', anonymous = False)
    try:
        ofvc(0)
    except rospy.ROSInterruptException:
        pass
