import sys
import os
# add the path to the folder that contains the AirSimClient module
sys.path += ["C:\AirSim\PythonClient"]

#import setup_path 
import airsim

import numpy as np
import os
import tempfile
import pprint
import cv2
import time

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

scale = 0.8
flag = False

#get prior features
img = cv2.imread('./prior_descriptor_patch_3.png')
#img = cv2.resize(img, (0,0), fx=scale, fy=scale)

# Convert BGR to HSV and parse HSV
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
img=h
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img2 = cv2.imread('./prior_descriptor_5.png')
img2 = cv2.resize(img2, (0,0), fx=scale, fy=scale)

# Convert BGR to HSV and parse HSV
hsv_img = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
img2=h

#img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

vf = cv2.xfeatures2d.SURF_create()
prior_keypoints, prior_descriptors = vf.detectAndCompute(img2, None)
#prior_keypoints2, prior_descriptors2 = vf.detectAndCompute(img2, None)

#airsim.wait_key('Press any key to takeoff')
print("Take off")
client.takeoffAsync().join()

#airsim.wait_key('Press any key to rotate vehicle')
print("Yaw 90")
client.rotateToYawAsync(90,5,1).join()


state = client.getMultirotorState()
s = pprint.pformat(state)
print("state: %s" % s)

# first inspection point
print("Start inspection")

moving_step = 1
acum_x = 0
acum_y = 0
acum_z = 3 #-5
print("{} {} {}".format(acum_x, acum_y, acum_z))


while True:
    flag = False

    result = client.simGetImage("0", airsim.ImageType.Scene)
    rawImage = np.fromstring(result, np.int8)
    png = cv2.imdecode(rawImage, cv2.IMREAD_UNCHANGED)
    
    # Convert BGR to HSV and parse HSV
    hsv_img = cv2.cvtColor(png, cv2.COLOR_BGR2HSV)
    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
    gray=h
    #gray = cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (0,0), fx=scale, fy=scale)
    
    '''
    # match from large view
    #prior check matching rate to stop forward
    keypoints, descriptors = vf.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = bf.match(descriptors,prior_descriptors)
    
    if not matches:
        print("force to move")
        client.moveToPositionAsync(0, 1, 0, 1).join()
    else:
    
        match_rate = len(matches) #*(1/len(descriptors) + 1/len(prior_descriptors))/2.0

        if (match_rate > 10):
            print("Too close")
            break
        
        else:
            #print("Continue inspection")
    '''
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
            match_rate = len(matches)#*(1/len(descriptors) + 1/len(prior_descriptors))/2.0
        mat_rate.append(match_rate)
        
    #print("[{:2.2f} {:2.2f} {:2.2f} {:2.2f}]".format(mat_rate[0], mat_rate[1], mat_rate[2], mat_rate[3]))
    print(mat_rate)
    print(sum(mat_rate))
    
    #for m in mat_rate:
    if sum(mat_rate) > 55:
            flag = True
    
    if flag:
        break
    
    #make decision to go next
    #direction = np.argmax(mat_rate)
    direction = np.argwhere(mat_rate == np.amax(mat_rate))
    direction = direction.flatten()
    
    if len(direction) > 1:
        #print("pick random move")
        direction = direction[np.random.randint(low=0, high=len(direction)-1)]
    else:
        direction = direction[0]
    '''
    if direction == 0:
        print("go 0")
        acum_x += moving_step  #left, forward, top
        acum_y += moving_step 
        acum_z -= moving_step
        client.moveToPositionAsync(acum_x, acum_y, acum_z, 1).join()
        #client.moveToPositionAsync(acum_x + moving_step, acum_y + moving_step, acum_z - moving_step, 1).join() #left, forward, top
    elif direction == 1:
        print("go 1")
        acum_x -= moving_step  #right, forward, top
        acum_y += moving_step
        acum_z -= moving_step
        client.moveToPositionAsync(acum_x, acum_y, acum_z, 1).join()
        #client.moveToPositionAsync(acum_x + -1*moving_step, acum_y + moving_step, acum_z + -1*moving_step, 1).join() #right, forward, top
    elif direction == 2:
        print("go 2")
        acum_x += moving_step  #left, forward, bottom
        acum_y += moving_step
        acum_z += moving_step
        client.moveToPositionAsync(acum_x, acum_y, acum_z, 1).join()
        #client.moveToPositionAsync(acum_x + moving_step, acum_y + moving_step, acum_z +moving_step, 1).join() #left, forward, bottom
    elif direction == 3:
        print("go 3")
        acum_x -= moving_step  #right, forward, bottom
        acum_y += moving_step
        acum_z += moving_step
        client.moveToPositionAsync(acum_x, acum_y, acum_z, 1).join()
        #client.moveToPositionAsync(acum_x + -1*moving_step, acum_y + moving_step, acum_z +moving_step, 1).join() #right, forward, bottom
    '''
    
    print("go to {}".format(direction))
    # forward
    acum_y += moving_step
    
    # left or right
    if direction == 0 or direction == 3 or direction == 6:
        #print("left")
        acum_x += moving_step
    elif direction == 2 or direction == 5 or direction == 8:
        #print("right")
        acum_x -= moving_step
    
    # up or bottom
    if direction == 0 or direction == 1 or direction == 2:
        #print("up")
        acum_z -= moving_step
    elif direction == 6 or direction == 7 or direction == 8:
        #print("down")
        acum_z += moving_step
    client.moveToPositionAsync(acum_x, acum_y, acum_z, 1).join()
        
print("start detection")
for i in range(10):
    acum_x -= moving_step
    client.moveToPositionAsync(acum_x, acum_y, acum_z, 1).join()
    time.sleep( 5 )
    print("no instances found")
'''
#airsim.wait_key('Press any key to move vehicle')
print("Starting position arrived")
speed = 1
distance = 9
moving = -8
client.moveToPositionAsync(0, distance, 0, speed).join()
client.moveToPositionAsync(0, distance, moving, speed).join()
client.moveToPositionAsync(-3, distance, moving, speed).join()
client.moveToPositionAsync(-3, distance, 0, speed).join()
client.moveToPositionAsync(-5, distance, 0, speed).join()
client.moveToPositionAsync(-5, distance, moving, speed).join()
client.moveToPositionAsync(-7, distance, moving, speed).join()
client.moveToPositionAsync(-7, distance, 0, speed).join()
client.moveToPositionAsync(-9, distance, 0, speed).join()
client.moveToPositionAsync(-9, distance, moving, speed).join()
client.moveToPositionAsync(-11, distance, moving, speed).join
client.moveToPositionAsync(-11, distance, 0, speed).join()
'''

print("Finished")
client.hoverAsync().join()
client.moveToPositionAsync(0, 0, 5, 1).join()

#airsim.wait_key('Press any key to reset to original state')
print("Reset")
client.armDisarm(False)
client.reset()

# that's enough fun for now. let's quit cleanly
client.enableApiControl(False)