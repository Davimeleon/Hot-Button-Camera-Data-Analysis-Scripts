#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 12:41:49 2023

@author: david
"""

file = "/Users/david/Documents/Programming_Projects/CaltechProject/Files/mouseTemp_Recordings/pdyn35/35L_body-05082024114752-0000.avi"


data_list = [
    {
        "Name":"VMHvl",
        "Number":"117",
        "Vid_Labels":[],
        "Vid_Paths":[],
        "Body_Vid_Path": ["F:/NPY2R/Males/117R/Face experiments/Laser stim/body-05222023134454-0000.avi"],
        "Stim_Path":"117_stim_init.npy",
    },
    {
        "Name":"VMHvl",
        "Number":"118",
        "Vid_Labels":[],
        "Vid_Paths":[],
        "Body_Vid_Path": ["F:/NPY2R/Males/118L/Face/experiment/body-05222023130132-0000.avi","F:/NPY2R/Males/118L/Face/experiment/body-05222023131253-0000.avi"],
        "Stim_Path":"118_stim_init.npy",
    },
    {
        "Name":"VMHvl",
        "Number":"121",
        "Vid_Labels":[],
        "Vid_Paths":[],
        "Body_Vid_Path": ["F:/NPY2R/Males/121L/Face experiments/Laser stim/body-05232023103609-0000.avi"],
        "Stim_Path":"121_stim_init.npy",
    },
    {
        "Name":"VMHvl",
        "Number":"122",
        "Vid_Labels":[],
        "Vid_Paths":[],
        "Body_Vid_Path": ["F:/NPY2R/Males/122RL/Face experiment/Laser Stim/body-05222023154401-0000.avi"],
        "Stim_Path":"122_stim_init.npy",
    },
    {
        "Name":"VMHvl",
        "Number":"126",
        "Vid_Labels":[],
        "Vid_Paths":[],
        "Body_Vid_Path": [file],
        "Stim_Path":"126_stim_init.npy",
    },
    {
        "Name":"VMHvl",
        "Number":"127",
        "Vid_Labels":[],
        "Vid_Paths":[],
        "Body_Vid_Path": ["F:/NPY2R/Males/127L/face experiment/Face laser stim/body-05222023142015-0000.avi"],
        "Stim_Path":"127_stim_init.npy",
    },
]

data = data_list[4]

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt


# read video



#VARIABLES
#
#

displayFrames = False

x1 = 25
widthVal = 45

y1 = 500
heightVal = 80

#
#
#



cap = cv2.VideoCapture(data["Body_Vid_Path"][0])

numFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Num Frames: " + str(numFrames))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Frame Width: {frame_width}")
print(f"Frame Height: {frame_height}")



def display_frame_with_rectangle(frame, sum_values, frameNum):
    # Draw a red rectangle around the region of interest
    top_left = (x1, y1)
    width = widthVal
    height = heightVal
    rect = Rectangle(top_left, width, height, linewidth=2, edgecolor='r', facecolor='none')

    # Display the frame with the rectangle and the calculated sum
    plt.figure()
    plt.imshow(frame)
    current_ax = plt.gca()
    current_ax.add_patch(rect)

    plt.text(50, 30, f"Sum: {sum_values}", color='r', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    plt.text(375, 30, f"Frame: {frameNum}", color='r', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    plt.show()



# Image #1
frameNumber = 1900

cap.set(cv2.CAP_PROP_POS_FRAMES, frameNumber)
ret, frame = cap.read()
#sum_values = np.sum(frame[680:690, 75:110])
sum_values = np.sum(frame[y1:(y1+heightVal), x1:((x1 + widthVal))])
display_frame_with_rectangle(frame, sum_values, frameNumber)


print(cap.isOpened())
print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(cap.get(cv2.CAP_PROP_POS_FRAMES))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
'''

#[613, 774, 2895, 3081, 5594, 5765, 7852, 8003, 10170, 10329, 12096, 12278, 14573, 14728, 16661, 16878]

# Image #2
cap.set(cv2.CAP_PROP_POS_FRAMES, 3500)
ret, frame2 = cap.read()
sum_values2 = np.sum(frame[680:699, 75:110])
display_frame_with_rectangle(frame2, sum_values2)

'''

cap.release()

cap = cv2.VideoCapture(data["Body_Vid_Path"][0])

light_on = []
lastFrame = 0
counter = 0

current_frame = 0

while ret:
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()
    #sum_values = np.sum(frame[570:650, 70:115])
    sum_values = np.sum(frame[y1:(y1+heightVal), x1:((x1 + widthVal))])
    
    #display_frame_with_rectangle(frame, sum_values, current_frame)
    
    if not ret or frame is None:  # Break the loop when there are no more frames to read
        break
    
    
    if (displayFrames):
        display_frame_with_rectangle(frame, sum_values, current_frame)
        #print(f"Frame Number: {current_frame}, Sum: {sum_values}")
    
    if sum_values > 400000:
        #frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        if (current_frame - lastFrame > 700):
            print('\n' * 2)
            
        if (counter == 0):
            print(f"Frame Number: {current_frame}, Sum: {sum_values}")
            display_frame_with_rectangle(frame, sum_values, current_frame)
        
        if (counter > 0):
            counter -= 1
        
        if (current_frame - lastFrame > 700):
            counter = 587
            
        lastFrame = current_frame
    
    light_on.append(sum_values)
    
    current_frame += 1
    #print(current_frame)
    


plt.plot(light_on)

cutoff = 150000
plt.axhline(y=cutoff, color="y", linestyle="-")

temp = np.array(light_on) > cutoff
stim_start = []
for i, e in enumerate(temp):
    if (e and not temp[(i-300):i].any()):
        stim_start.append(i)
        plt.axvline(x=i, color="r", linestyle="-")
        #print(i)

with open(data["Stim_Path"], "wb") as f:
    np.save(f, stim_start)
    
print("Code is done running.")


