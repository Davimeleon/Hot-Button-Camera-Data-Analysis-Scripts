#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 20:09:50 2024

@author: david
"""

import numpy as np
import struct
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import wilcoxon


numFramesAfter = 30
sumBaseline = 0
numBaseline = 0
sumInteraction = 0
numInteraction = 0


currentBaseline = 0
differences = []
temperatureAfterInteraction = []
xArr = []


class Mouse:
    def __init__(self, binFile, mouseName, insertFrame, exitFrame):
        #Given Info
        self.binFile = binFile
        self.insertFrame = insertFrame
        self.exitFrame = exitFrame
        
        #Data Vars
        self.avgTempAboveThreshhold = []
        self.avgTempAboveThreshhold100Frames = []
        self.avgTempAboveThreshholdNoWall = []
        self.timeDifferences = []   #Stores timeDifference between each frame
        self.timeDifferencesPrefixSum = []
        
        self.averageBaseline = 0
        
        self.img = 0 #Thermograph Image Array
        self.mouseName = mouseName
        
        #Other Vars
        self.width = 0
        self.height = 0
        self.nframes = 0
        
        self.outputPath = "/Users/david/Documents/Programming_Projects/CaltechProject/VideoImages/aggressionFile/"
        self.temperatureThreshold = 25
        
    
    def openFile(self):
       counter = 0
       with open(self.binFile, 'rb') as file:
            # Read size of file in bytes
            size_in_bytes = file.seek(0, 2)
        
            # Move back to the beginning of the file
            file.seek(0)
        
            # Loop to read all frames
            ii = 0
            
            while True:
                self.width = struct.unpack('H', file.read(2))[0] 
                self.height = struct.unpack('H', file.read(2))[0]
                
                if ii == 0:
                    self.nframes = size_in_bytes // (148 + 4 * self.width * self.height)
                    self.img = np.zeros((self.height, self.width, self.nframes), dtype=np.float32)
                    t = np.zeros(self.nframes, dtype=np.uint64)
                    print("width: " + str(self.width))
                    print("height: " + str(self.height))
                
                
                t[ii] = struct.unpack('Q', file.read(8))[0]
        
                counter += 1
            
                binary_data = file.read(self.width * self.height * 4)
                self.img[:, :, ii] = np.frombuffer(binary_data, dtype=np.float32).reshape(self.height, self.width)
                
                ii += 1
                
                # Break the loop if ii exceeds the calculated number of frames
                if ii >= self.nframes:
                    break
    
    def makeTemperatureGraph(self):
        global sumBaseline, sumInteraction, numBaseline, numInteraction, currentBaseline, differences, xArr
        
        for frameIndex in range (self.nframes):
            frame = self.img[:, :, frameIndex]
            above_threshold = frame[frame > self.temperatureThreshold]
            
            #plt.imshow(self.img[:, :, frameIndex], cmap='hot', vmin= 25, vmax=36)
            #plt.show()
            
            if len(above_threshold) > 0:
                avg_temp_above_threshold = np.mean(above_threshold)
                #avg_temp_above_threshold = np.max(above_threshold)
                self.avgTempAboveThreshhold.append(avg_temp_above_threshold)
                
                
                # Find center of mass
                y, x = np.indices(frame.shape)
                com_x = np.sum(x * (frame > self.temperatureThreshold)) / np.sum(frame > self.temperatureThreshold)
                com_y = np.sum(y * (frame > self.temperatureThreshold)) / np.sum(frame > self.temperatureThreshold)
                
                #print("y: " + str(com_y))
                #print("x: " + str(com_x))
    
                # Check if center of mass is near the wall
                near_wall = com_x < 30 or com_x > frame.shape[1] - 30 or com_y < 30 or com_y > frame.shape[0] - 30
    
                if not near_wall:
                    self.avgTempAboveThreshholdNoWall.append(avg_temp_above_threshold)
        
        
        self.avgTempAboveThreshhold = self.avgTempAboveThreshholdNoWall

        self.avgTempAboveThreshhold100Frames = self.avgTempAboveThreshhold[self.insertFrame + 3: self.insertFrame + numFramesAfter + 3]
        
        #sem = np.std(self.avgTempAboveThreshhold) / np.sqrt(len(self.avgTempAboveThreshhold))
        
        mean = np.mean(self.avgTempAboveThreshholdNoWall)
        std = np.std(self.avgTempAboveThreshholdNoWall)
        sem = std/np.sqrt(len(self.avgTempAboveThreshholdNoWall))
        
        
        #sem = np.nanstd(self.avgTempAboveThreshholdNoWall, ddof=1) / np.sqrt(len(self.avgTempAboveThreshholdNoWall))
        print("SEM is: " + str(sem))


        # Plotting the data
        plt.plot(self.avgTempAboveThreshholdNoWall, label = "Avg Temp Mouse")
        #plt.errorbar(range(len(self.avgTempAboveThreshholdNoWall)), self.avgTempAboveThreshholdNoWall, yerr = sem, fmt='o')
        #plt.fill_between(range(len(self.avgTempAboveThreshholdNoWall)), self.avgTempAboveThreshholdNoWall - sem, self.avgTempAboveThreshholdNoWall + sem, alpha=0.2, label='SEM', color='gray')
        plt.xlabel('Frame Number (≈ seconds)')
        plt.ylabel("Avg Temp Mouse")
        plt.title("Temperature vs. Time")
        plt.legend()
        plt.show()

        #Add Red Lines
        #plt.axvline(x=framesBefore, color='red', linestyle='dashed', alpha=0.5)
        #plt.axvline(x=max_length - framesAfter, color='red', linestyle='dashed', alpha=0.5)
        
        
        if (self.mouseName != "Baseline"):
            avg_temp_before_insert = np.mean(self.avgTempAboveThreshhold[:(self.insertFrame-1)])
            avg_temp_after_exit = np.mean(self.avgTempAboveThreshhold[(self.exitFrame+1):(self.exitFrame+numFramesAfter+1)])
            sumInteraction += avg_temp_after_exit
            numInteraction += 1
            
            temperatureAfterInteraction.append(avg_temp_after_exit)
            differences.append((avg_temp_after_exit - currentBaseline))
            xArr.append(np.random.uniform(-0.1, 0.1))
            
            #print(f"Average Temp Before Intruder: {avg_temp_before_insert}")
            print(f"Average Temp After Intruder: {avg_temp_after_exit}")
            
            #plt.bar(["Before Insert", "After Exit"], [avg_temp_before_insert, avg_temp_after_exit])
            #plt.ylim(25, 30)
            #plt.ylabel("Average Temp Mouse")
            #plt.title("Average Temperature Before Insert vs. After Exit")
            #plt.show()
        else:
            avgTempOverall = np.mean(self.avgTempAboveThreshhold)
            sumBaseline += avgTempOverall
            currentBaseline = avgTempOverall
            self.averageBaseline = currentBaseline
            numBaseline += 1
            print(f"The Average Baseline Temperature was: {avgTempOverall}")
    
    
    def output(self):
        for i in range(self.nframes): 
            #Preset Limits
            #plt.imshow(self.img[:, :, i], cmap='hot', interpolation='nearest')
            plt.imshow(self.img[:, :, i], cmap='hot', vmin= self.temperatureThreshold, vmax=35)
            plt.text(245, 30, f"Frame: {i}", color='r', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
            
            name = self.outputPath + f'{self.mouseName}_%05d.png' % i
            plt.savefig(name)
            
            plt.show()
            


def averageoOfAvgTempAboveThreshold(mouseObjects):
    avg_all = np.zeros_like(mouseObjects[0].avgTempAboveThreshhold100Frames)
    
    for mouse in mouseObjects:
        avg_all += mouse.avgTempAboveThreshhold100Frames
        
    avg_all /= len(mouseObjects)
        
    #num = (mouseObjects[0].avgTempAboveThreshhold100Frames[0] + mouseObjects[1].avgTempAboveThreshhold100Frames[0] + mouseObjects[2].avgTempAboveThreshhold100Frames[0] + mouseObjects[3].avgTempAboveThreshhold100Frames[0])/4      
    #print("Test: " + str(num))
    #print ("Avg_all: " + str(avg_all))
    
    #avg_all = np.mean(avg_all, axis=0)    
    
    plt.plot(avg_all, label = "Avg Temp Mouse")
    plt.xlabel('Frame Number (≈ seconds)')
    plt.ylabel("Avg Temp Mouse")
    plt.title("Avg Temp Mouse over Time")
    plt.legend()
    plt.show()
    
    


           

'''

#Mouse 226
#
#

#Aggression Baseline
print("BASELINE M226")
binFile = "/Users/david/Documents/Programming_Projects/CaltechProject/Files/mouseTemp_Recordings/mouse226_natural_aggression/226_base-2024-05-01_11-23-56.bin"
mouseBaseline226 = Mouse(binFile, "Baseline", 100, 105)
mouseBaseline226.openFile()
mouseBaseline226.makeTemperatureGraph()
#mouseBaseline226.output()
print('\n')


#Aggression Male 1
print("MALE 1: M226")
binFile = "/Users/david/Documents/Programming_Projects/CaltechProject/Files/mouseTemp_Recordings/mouse226_natural_aggression/226_male1-2024-05-01_11-34-33.bin"
male1Interaction226 = Mouse(binFile, "male1_226", 32, 202)
male1Interaction226.openFile()
male1Interaction226.makeTemperatureGraph()
#male1Interaction226.output()
print('\n')


#Aggression Male 2
print("MALE 2: M226")
binFile = "/Users/david/Documents/Programming_Projects/CaltechProject/Files/mouseTemp_Recordings/mouse226_natural_aggression/226_male2-2024-05-01_11-44-04.bin"
male2Interaction226 = Mouse(binFile, "male2_226", 82, 333)
male2Interaction226.openFile()
male2Interaction226.makeTemperatureGraph()
#male2Interaction226.output()
print('\n')


#Aggression Male 3
print("MALE 3: M226")
binFile = "/Users/david/Documents/Programming_Projects/CaltechProject/Files/mouseTemp_Recordings/mouse226_natural_aggression/226_male3-2024-05-01_11-54-56.bin"
male3Interaction226 = Mouse(binFile, "male3_226", 180, 250)
male3Interaction226.openFile()
male3Interaction226.makeTemperatureGraph()
#male3Interaction226.output()
print('\n')

aggressionInteractions = [male1Interaction226, male2Interaction226, male3Interaction226]
averageoOfAvgTempAboveThreshold(aggressionInteractions)

#
#
#

'''


#Mouse 239
#
#

#Aggression Baseline
print("BASELINE M239")
binFile = "/Users/david/Documents/Programming_Projects/CaltechProject/Files/mouseTemp_Recordings/mouse239_natural_aggression/239L_base-2024-05-01_09-32-37.bin"
mouseBaseline239 = Mouse(binFile, "Baseline", 100, 105)
mouseBaseline239.openFile()
mouseBaseline239.makeTemperatureGraph()
#mouseBaseline239.output()
print('\n')


#Aggression Male 1
print("MALE 1: M239")
binFile = "/Users/david/Documents/Programming_Projects/CaltechProject/Files/mouseTemp_Recordings/mouse239_natural_aggression/239L_male1-2024-05-01_09-59-34.bin"
male1Interaction239 = Mouse(binFile, "male1", 68, 426)
male1Interaction239.openFile()
male1Interaction239.makeTemperatureGraph()
#male1Interaction239.output()
print('\n')


#Aggression Male 2
print("MALE 2: M239")
binFile = "/Users/david/Documents/Programming_Projects/CaltechProject/Files/mouseTemp_Recordings/mouse239_natural_aggression/239L_male2-2024-05-01_10-12-37.bin"
male2Interaction239 = Mouse(binFile, "male2", 67, 105)
male2Interaction239.openFile()
male2Interaction239.makeTemperatureGraph()
#male2Interaction239.output()
print('\n')


#Aggression Male 3
print("MALE 3: M239")
binFile = "/Users/david/Documents/Programming_Projects/CaltechProject/Files/mouseTemp_Recordings/mouse239_natural_aggression/239L_Male3-2024-05-01_10-18-59.bin"
male3Interaction239 = Mouse(binFile, "male3", 76, 119)
male3Interaction239.openFile()
male3Interaction239.makeTemperatureGraph()
#male3Interaction239.output()
print('\n')

aggressionInteractions = [male1Interaction239, male2Interaction239, male3Interaction239]
averageoOfAvgTempAboveThreshold(aggressionInteractions)

#
#
#




#Mouse 1
#
#

#Aggression Baseline
print("BASELINE")
binFile = "/Users/david/Documents/Programming_Projects/CaltechProject/Files/mouseTemp_Recordings/naturalAggressionTrials/m1_baseline-2024-04-26_17-05-51.bin"
mouseBaseline = Mouse(binFile, "Baseline", 100, 105)
mouseBaseline.openFile()
mouseBaseline.makeTemperatureGraph()
#mouseBaseline.output()
print('\n')


#Aggression Male 1
print("MALE 1")
binFile = "/Users/david/Documents/Programming_Projects/CaltechProject/Files/mouseTemp_Recordings/naturalAggressionTrials/m1_male-2024-04-26_17-14-46.bin"
male1Interaction = Mouse(binFile, "male1", 15, 90)
male1Interaction.openFile()
male1Interaction.makeTemperatureGraph()
#male1Interaction.output()
print('\n')


#Aggression Male 2
print("MALE 2")
binFile = "/Users/david/Documents/Programming_Projects/CaltechProject/Files/mouseTemp_Recordings/naturalAggressionTrials/m1_male2-2024-04-26_17-22-43.bin"
male2Interaction = Mouse(binFile, "male2", 21, 87)
male2Interaction.openFile()
male2Interaction.makeTemperatureGraph()
#male2Interaction.output()
print('\n')


#Aggression Male 3
print("MALE 3")
binFile = "/Users/david/Documents/Programming_Projects/CaltechProject/Files/mouseTemp_Recordings/naturalAggressionTrials/m1_male3-2024-04-26_17-31-57.bin"
male3Interaction = Mouse(binFile, "male3", 59, 120)
male3Interaction.openFile()
male3Interaction.makeTemperatureGraph()
#male3Interaction.output()
print('\n')


#Aggression Male 4
#print("MALE 4")
#binFile = "/Users/david/Documents/Programming_Projects/CaltechProject/Files/mouseTemp_Recordings/naturalAggressionTrials/m1_male4-2024-04-26_17-39-58.bin"
#male4Interaction = Mouse(binFile, "male4", 110, 152)
#male4Interaction.openFile()
#male4Interaction.makeTemperatureGraph()
#male4Interaction.output()
#print('\n')

aggressionInteractions = [male1Interaction, male2Interaction, male3Interaction]
averageoOfAvgTempAboveThreshold(aggressionInteractions)

#
#
#



#Analysis



#Paired Plot
#
#

#before_list = [mouseBaseline226.averageBaseline] * 3 + [mouseBaseline239.averageBaseline] * 3 + [mouseBaseline.averageBaseline] * 3
before_list = [mouseBaseline239.averageBaseline] * 3 + [mouseBaseline.averageBaseline] * 3
after_list = temperatureAfterInteraction

legend_labels = []
colors = ['lightblue', 'lightgreen', 'lightcoral']
offset = 0.02

beforeAvg = np.mean(before_list)
afterAvg = np.mean(after_list)

avg_temp_diff = (afterAvg - beforeAvg)

assert len(before_list) == len(after_list), "The two lists must have the same length"

for idx in range(0, len(before_list), 3):
    color_idx = (idx // 3) % len(colors)
    line = plt.plot(['Baseline', 'After Bout'], [before_list[idx:idx+3], after_list[idx:idx+3]], marker='o', linestyle='-', color=colors[color_idx], alpha=0.65, linewidth=1)
    legend_labels.append(line[0])

# Plot average line
avgLine = plt.plot([0, 1], [beforeAvg, afterAvg], color='black', linestyle='--', linewidth=3.5, marker='o', zorder=5, label=f'Average Temperature Increase\n(Temperature Increase: {avg_temp_diff:.4f})')
legend_labels.append(avgLine[0])

# Set labels and title
plt.ylabel("Temperature (°C)")

# Adjust the subplot parameters
plt.subplots_adjust(left=0.1, right=0.4, top=0.9, bottom=0.1)

# Add the legend with specified number of labels
plt.legend(handles=legend_labels, loc='upper left', bbox_to_anchor=(1, 1))

# Perform statistical tests
'''t_stat, p_value = ttest_rel(before_list, after_list)
print(f"Paired t-test p-value: {p_value}")'''


difference = [x - y for x, y in zip(after_list, before_list)]
print("Performing Tests")
print(stats.kruskal(before_list, after_list))
print(stats.ttest_rel(before_list, after_list))
res = wilcoxon(difference)
print("Wilcoxon p-value: " + str(res.pvalue))

p_value = res.pvalue

# Determine significance text
if p_value < 0.001:
    significance_text = '***'
elif p_value < 0.01:
    significance_text = '**'
elif p_value < 0.05:
    significance_text = '*'
else:
    significance_text = 'ns'

# Position the text within the plot area
plt.text(1.9, beforeAvg + 0.1, "Wilcoxon Test\n", fontsize=21, color='black', fontweight="bold", ha='center')
plt.text(1.9, beforeAvg - 0.05, f'$p = {p_value:.5f}$\n{significance_text}', fontsize=15, color='black', ha='center')

# Draw the plot
plt.draw()

# Save the figure
name = "/Users/david/Documents/Programming_Projects/CaltechProject/VideoImages/savePlots/naturalAggressionTemp_pairedPlot_no226.svg"
plt.savefig(name, format='svg')

plt.show()

#
#
#




print("Differences: " + str(differences))

average_diff = np.mean(differences)
sem_diff = np.std(differences) / np.sqrt(len(differences))

# Plotting the bar chart
plt.figure(figsize=(2, 6))
plt.bar([''], [average_diff], yerr=[sem_diff], capsize=5, color='blue', label='Temp Increase')
plt.errorbar([''], [average_diff], yerr=[sem_diff], fmt='o', color='black')

# Adding individual points
plt.scatter(xArr, differences, color='orange', label='Individual Trials')

# Adding labels and title
plt.ylim(-0.2, 0.7)
plt.xlabel('Temperature Increase', fontsize=14)
plt.ylabel('Average Temperature (°C)', fontsize=14)
plt.title('Natural Aggression Temperature Increase', fontsize=18, y = 1.05)
plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y')
plt.show()




averageBaseline = (sumBaseline)/(numBaseline)
averageInteraction = (sumInteraction)/(numInteraction) 

print('\n' * 2)
print("Average Baseline is:  " + str(averageBaseline) + '\n')
print("Average Interaction is:  " + str(averageInteraction))

print('\n' * 2 + "Difference is: " + str((averageInteraction - averageBaseline)))



categories = ['Baseline', 'Interaction']
values = [averageBaseline, averageInteraction]

# Plotting the bar chart
plt.bar(categories, values, color=['blue', 'green'])
plt.ylim(27.5, 27.8)
#plt.xlabel('Condition')
plt.ylabel('Average Temperature')
plt.title('Natural Aggression Temperature Difference')
plt.show()




