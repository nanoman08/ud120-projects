# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 10:10:47 2016

@author: CHOU_H
"""

# -----------------
#circular motion
# USER INSTRUCTIONS
#
# Write a function in the class robot called move()
#
# that takes self and a motion vector (this
# motion vector contains a steering* angle and a
# distance) as input and returns an instance of the class
# robot with the appropriate x, y, and orientation
# for the given motion.
#
# *steering is defined in the video
# which accompanies this problem.
#
# For now, please do NOT add noise to your move function.
#
# Please do not modify anything except where indicated
# below.
#
# There are test cases which you are free to use at the
# bottom. If you uncomment them for testing, make sure you
# re-comment them before you submit.

from math import *
import random
# --------
# 
# the "world" has 4 landmarks.
# the robot's initial coordinates are somewhere in the square
# represented by the landmarks.
#
# NOTE: Landmark coordinates are given in (y, x) form and NOT
# in the traditional (x, y) format!

landmarks  = [[0.0, 100.0], [0.0, 0.0], [100.0, 0.0], [100.0, 100.0]] # position of 4 landmarks
world_size = 100.0 # world is NOT cyclic. Robot is allowed to travel "out of bounds"
max_steering_angle = pi/4 # You don't need to use this value, but it is good to keep in mind the limitations of a real car.

# ------------------------------------------------
# 
# this is the robot class
# x, y, orienation, length, bearing_noise, steering_noise
# __repr_
# set
# set_noise
# move

class robot:
    
    def __init__(self, length = 10):
        self.x = random.random()*world_size
        self.y = random.random()*world_size
        self.orientation = random.random()*2*pi
        self.length = length
        self.bearing_noise = 0
        self.steering_noise = 0
        self.distance_noise = 0
        self.landmarks = [[0.0, 100.0], [0.0, 0.0], [100.0, 0.0], [100.0, 100.0]]
    
    def __repr__(self):
        #return '[x={:.6} y={:.6} orient={:.6}]'.format(str(self.x), str(self.y), str(self.orientation))
        return '[x=%.6s y=%.6s orient=%.6s]' % (str(self.x), str(self.y), str(self.orientation))
        
    def set(self, new_x, new_y, new_orient):
        self.x = new_x
        self.y = new_y
        self.orientation = new_orient
        
    def set_noise(self, new_b_noise, new_s_noise, new_d_noise):
        self.bearing_noise = new_b_noise
        self.steering_noise = new_s_noise
        self.distance_noise = new_d_noise

    ############# ONLY ADD/MODIFY CODE BELOW HERE ###################

    # --------
    # move:
    #   move along a section of a circular path according to motion
    #   L*beta = d*tan(alpha)  alpha: steering angle, beta:resulting orientation change. 
    #   L: length between front/back wheels, d: distance moved, needs to assume it is small
    #   R = d/beta, Cx, Cy = x-Rsin(theta), y+R*cos(theta)
    #   x' = x-Rsin(theta)+Rsin(theta+beta)= x-Rsin(theta)+Rsin(theta)cos(beta)+Rsin(beta)cos(theta)
    #      ~ x + Rsin(beta)cos(theta) ~ x+dcos(theta)
    #   y' ~ y +dsin(theta)
    #   theta' = (theta +d*tan(alpha)/L) mod 2pi
    def move(self, motion): # Do not change the name of this function
        
    
        # ADD CODE HERE
        max_tolerance = 0.001
        result = self
        result.length = self.length
        result.bearing_noise  = self.bearing_noise
        result.steering_noise = self.steering_noise
        result.distance_noise = self.distance_noise
        
        
        distance = random.gauss(motion[1], result.distance_noise)
        steering = random.gauss(motion[0], result.steering_noise)
        beta = distance*tan(steering)/result.length 
        
        if abs(beta) <= max_tolerance:
            
            result.x = self.x + distance*cos(self.orientation)
            result.y = self.y + distance*sin(self.orientation)
            result.orientation = (self.orientation + beta) % (2*pi)
        
        else:
            cx = self.x-distance/(beta)*sin(self.orientation) 
            cy = self.y+distance/(beta)*cos(self.orientation)
            result.orientation = (self.orientation + beta) % (2*pi)
            result.x = cx + distance/beta*sin(result.orientation) 
            result.y = cy - distance/beta*cos(result.orientation)
        
        return result

#        return result # make sure your move function returns an instance
                      # of the robot class with the correct coordinates.

    def sense(self, add_noise = 1): #do not change the name of this function
        Z = []

        # ENTER CODE HERE
        for xy in self.landmarks:
            angle = atan2(xy[0]-self.y, xy[1]-self.x)-self.orientation
            if add_noise:
                angle += random.gauss(0.0, self.bearing_noise)
            
            Z.append(angle % (2*pi))
        # HINT: You will probably need to use the function atan2()

        return Z #Leave this line here. Return vector Z of 4 bearings.
    
                      
    ############## ONLY ADD/MODIFY CODE ABOVE HERE ####################
        

## IMPORTANT: You may uncomment the test cases below to test your code.
## But when you submit this code, your test cases MUST be commented
## out. Our testing program provides its own code for testing your
## move function with randomized motion data.

## --------
## TEST CASE:
## 
## 1) The following code should print:
##       Robot:     [x=0.0 y=0.0 orient=0.0]
##       Robot:     [x=10.0 y=0.0 orient=0.0]
##       Robot:     [x=19.861 y=1.4333 orient=0.2886]
##       Robot:     [x=39.034 y=7.1270 orient=0.2886]
##
##
length = 20.
bearing_noise  = 0.0
steering_noise = 0.0
distance_noise = 0.0

myrobot = robot(length)
myrobot.set(0.0, 0.0, 0.0)
myrobot.set_noise(bearing_noise, steering_noise, distance_noise)

motions = [[0.0, 10.0], [pi / 6.0, 10], [0.0, 20.0]]

T = len(motions)

print 'Robot:    ', myrobot
for t in range(T):
    myrobot = myrobot.move(motions[t])
    print 'Robot:    ', myrobot
##
##

## IMPORTANT: You may uncomment the test cases below to test your code.
## But when you submit this code, your test cases MUST be commented
## out. Our testing program provides its own code for testing your
## move function with randomized motion data.

    
## 2) The following code should print:
##      Robot:     [x=0.0 y=0.0 orient=0.0]
##      Robot:     [x=9.9828 y=0.5063 orient=0.1013]
##      Robot:     [x=19.863 y=2.0201 orient=0.2027]
##      Robot:     [x=29.539 y=4.5259 orient=0.3040]
##      Robot:     [x=38.913 y=7.9979 orient=0.4054]
##      Robot:     [x=47.887 y=12.400 orient=0.5067]
##      Robot:     [x=56.369 y=17.688 orient=0.6081]
##      Robot:     [x=64.273 y=23.807 orient=0.7094]
##      Robot:     [x=71.517 y=30.695 orient=0.8108]
##      Robot:     [x=78.027 y=38.280 orient=0.9121]
##      Robot:     [x=83.736 y=46.485 orient=1.0135]
##
##
##length = 20.
##bearing_noise  = 0.0
##steering_noise = 0.0
##distance_noise = 0.0
##
##myrobot = robot(length)
##myrobot.set(0.0, 0.0, 0.0)
##myrobot.set_noise(bearing_noise, steering_noise, distance_noise)
##
##motions = [[0.2, 10.] for row in range(10)]
##
##T = len(motions)
##
##print 'Robot:    ', myrobot
##for t in range(T):
##    myrobot = myrobot.move(motions[t])
##    print 'Robot:    ', myrobot

## IMPORTANT: You may uncomment the test cases below to test your code.
## But when you submit this code, your test cases MUST be commented
## out. Our testing program provides its own code for testing your
## move function with randomized motion data.