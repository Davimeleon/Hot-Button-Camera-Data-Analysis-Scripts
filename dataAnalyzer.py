#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import struct
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import sem
from scipy.signal import savgol_filter
from scipy.stats import wilcoxon
from scipy import stats
from sklearn.metrics import r2_score
import h5py
import cv2
import matplotlib.gridspec as gridspec
import pyqtgraph as pg

#Standard Constants
framesCutOff = 10
framesBefore = 30 #20
framesAfter = 75 #75
numberActivations = 8
#lowerLimit = 30.75
#upperLimit = 32.25
lowerLimit = 29.75
upperLimit = 31.75
baselineTemp = 25
showMouseTherm = False
smoothData = True
window_length = 13  # Must be an odd integer
poly_order = 1  # Polynomial order of the filter

nFrames = 0

labelY = 'Average Temperature (°C)'
#labelY = 'Proportion of Max Temperature'


class Mouse:
    def __init__(self, fileN, pupilPath, videoPath, outputP, imageN, frameN, row, col, nm, bodyPath, fps):
        #Input Vars
        self.filename = fileN
        self.pupilDataPath = pupilPath
        self.head_video_path = videoPath
        self.outputPath = outputP
        self.bodyPath = bodyPath
        self.imageName = imageN
        self.frameNumbersActual = frameN
        self.frameNumbers = [elem / fps * 1000 for elem in frameN] #self.frameNumbers = [elem / 30.35 * 1000 for elem in frameN]
        #Center Coords for 11x11 Box
        self.center_row = row
        self.center_col = col
        self.name = nm

        #Data Vars
        self.timeDifferences = []   #Stores timeDifference between each frame
        self.timeDifferencesPrefixSum = []
        self.averageTemp = []       #Stores the average temperature of each frame for the mouse (above a certain temperature)
        self.avgTempBox = []        #Stores the Average Temperature for each frame for the 11x11 Box
        self.maxTemp = []           #Stores the maximum temp of each frame
        self.maxTempBox = []
        self.lightOnFrames = []     #Stores the frame on which the light is activated
        self.all_plots_data = []    #Stores the data for each plot of each trial
        # Lists to store data for each trial (20 Frames)
        self.avg_temp_before_list = []
        self.avg_temp_during_list = []
        self.avg_temp_after_list = []
        self.avg_temp_before_during_after = 0
        #Arrays to Store Average Thermal Images
        self.average_pixels_before = 0
        self.average_pixels_during = 0
        self.average_pixels_after = 0
        
        #Average Temperature of Every Trial of the 20 frames before, 20 frames during, and 20 frames after
        self.avg_temp_before_avg = 0
        self.avg_temp_during_avg = 0
        self.avg_temp_after_avg = 0
        self.maxTempTrial = [] #Maximum Temperature for Each Trial
        
        self.img = 0 #Thermograph Image Array
        
        #PupilData
        self.pupil_areas = []
        self.cameraFrameNumbers = frameN
        self.avgPupilAreaAllTrials = []
        self.squarePupilAreasSum = []
        
        #Other Vars
        self.width = 0
        self.height = 0
        self.nframes = 0
        
        self.timeDiffCutoff = 0
        
        #Speed Stuff
        self.numSpeedFrames = 0
        self.numDuringFrames = 0
        self.numAfterFrames = 0
        self.avgSpeedDuring = 0
        self.avgSpeedAfter = 0
    
    
    def openFile(self):
        print("Frame Numbers: " + str(self.frameNumbers))
        
        #Vars
        counter = 0
        previousTime = 0
        counterFrame = 0
        counterLightOn = 0
        total_time_diff = 0
        
        accumulated_pixels_total = 0
        
        # Open file
        with open(self.filename, 'rb') as file:
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
                
                
                if (previousTime != 0):
                    time_diff = (int)((t[ii] - previousTime ) / 1000000) 
                    self.timeDifferences.append(time_diff)
                    total_time_diff += time_diff
                    
                    if (counterFrame < len(self.frameNumbers) and total_time_diff > self.frameNumbers[counterFrame]):
                        self.lightOnFrames.append(counter)
                        counterFrame += 1
                        
                
                #if (counterLightOn < len(self.frameNumbers) and total_time_diff > self.frameNumbers[counterLightOn] and total_time_diff < self.frameNumbers[counterLightOn + 1]):
                    #plt.text(0.5, 0.95, 'Laser Activated', transform=plt.gca().transAxes,
                             #ha='center', va='center', fontsize=16, fontweight='bold', color='red')
                
                if (counterLightOn < len(self.frameNumbers) and total_time_diff > self.frameNumbers[counterLightOn + 1]):
                    counterLightOn += 2
                    
                
                previousTime = t[ii]
                
                # Calculate average temperature for pixels with minimum temperature of 25 degrees Celsius
                avg_temp = np.mean(self.img[:, :, ii][self.img[:, :, ii] >= baselineTemp])
                self.averageTemp.append(avg_temp)
               
                # Calculate maximum temperature for each frame
                max_temp = np.max(self.img[:, :, ii])
                self.maxTemp.append(max_temp)
                
                row_start = max(center_row - 5, 0)
                row_end = min(center_row + 5, self.height)
                col_start = max(center_col - 5, 0)
                col_end = min(center_col + 5, self.width)
                
                # Make the Square
                square = patches.Rectangle((col_start, row_start), col_end - col_start, row_end - row_start, linewidth=0.5, edgecolor='blue', facecolor='none')
        
                # Extract the 5x5 box of pixels
                box_pixels = self.img[row_start:row_end+1, col_start:col_end+1, ii]
                
                # Calculate the average temperature of the box
                avg_temp_box = np.mean(box_pixels)
                #print("avg_temp_box: " + str(avg_temp_box))
                self.avgTempBox.append(avg_temp_box)
                
                max_temp_box = np.max(box_pixels)
                self.maxTempBox.append(max_temp_box)
                    
                #Find Max Temp
                accumulated_pixels_total += self.img[:, :, ii]
                
                if ((ii == framesCutOff and showMouseTherm == False) or (ii >= framesCutOff and showMouseTherm == True and ii <= framesCutOff + 400)):
                    # Add the square to the plot
                    plt.gca().add_patch(square)
                    
                    #Manual Limits
                    #plt.imshow(img[:, :, ii-1], cmap='hot', vmin=lowerLimit, vmax=upperLimit)
                    
                    #Preset Limits
                    plt.imshow(self.img[:, :, ii], cmap='hot', interpolation='nearest')
                    name = self.outputPath + f'{self.imageName}_%05d.png' % ii
                    
                    #plt.savefig(name)
                    
                    plt.show()
                
                if (ii == framesCutOff):
                    self.timeDiffCutoff = total_time_diff
                
                ii += 1
                if ii >= self.nframes:
                    break
            
            #Pupil Data
            data = np.load(self.pupilDataPath, allow_pickle=True).item()
            
            # Extract pupil data 
            pupil_data = data.get('pupil', [])

            # Extracting 'area' values from the pupil data and storing them in a list
            pupil_list = [pupil_frame.get('area', 0) for pupil_frame in pupil_data]
            self.pupil_areas = pupil_list[0]
            
            
            #Cutoff first 60 frames
            print("Len avgTempBox is " + str(len(self.avgTempBox)))
            self.avgTempBox = self.avgTempBox[framesCutOff:]
            
            self.averageTemp = self.averageTemp[framesCutOff:]
            self.maxTemp = self.maxTemp[framesCutOff:]
            
            #print("timeDifferences: " + str(self.timeDifferences))
            
            iii = 1
            self.timeDifferencesPrefixSum.append(self.timeDifferences[0])
            for num in self.timeDifferences[1:]:
                cumulative_sum = self.timeDifferencesPrefixSum[iii-1] + num
                self.timeDifferencesPrefixSum.append(cumulative_sum)
                iii+=1
            
            #print("time difference prefix sum: " + str(self.timeDifferencesPrefixSum))
            
            #SMOOTHING FILTER ON OR OFF
            if (smoothData):
                self.avgTempBox = savgol_filter(self.avgTempBox, window_length, poly_order)
            
            
            totalTime = 0
            for time in self.timeDifferences: 
                totalTime += time
            
            print("Total Time: " + str(totalTime))
            
            #print("timeDifferences: " + str(self.timeDifferences))
            #self.timeDifferences = self.timeDifferences[framesCutOff:]
            
            #Find Coords of Max Temp
            if (center_row == 0 and center_col == 0):
                desired_area = accumulated_pixels_total[:, 100:300]

                # Find the index of the maximum value in the flattened array
                max_index_flattened = np.argmax(desired_area)       
                # Convert the flattened index to row and column indices using unravel_index
                self.center_row, self.center_col = np.unravel_index(max_index_flattened, accumulated_pixels_total.shape)
                print("center_row: " + str(self.center_row))
                print("center_col: " + str(self.center_col))
                
    
    def getData(self):
        doneFrame = False
        tempIndex = 0

        totalTempBefore = 0
        totalTempDuring = 0
        totalFramesDuring = 0
        totalTempAfter = 0
        totalTempAfter20 = 0
        
        #Initialize Thermal Image Arrays
        accumulated_pixels_before = np.zeros((self.height, self.width))
        accumulated_pixels_during = np.zeros((self.height, self.width))
        accumulated_pixels_after = np.zeros((self.height, self.width))
        
        
        indexTrial = 0
        for frame in self.lightOnFrames:
            if (doneFrame):
                maxTemp = 0
                doneFrame = False
                index = int(frame) - framesCutOff - 1
                #print("Index is " +  str(index))
                
                
                #Average Temp Before Laser Activation
                avgTempBefore = 0
                for i in range (tempIndex-framesBefore, tempIndex):
                    avgTempBefore += self.avgTempBox[i]
                    # Accumulate pixel values for each frame
                    accumulated_pixels_before += self.img[:, :, i + framesCutOff]
                    
                totalTempBefore += avgTempBefore
                avgTempBefore /= framesBefore
                self.avg_temp_before_list.append(avgTempBefore)
                
                #print("The average temperature for the 20 frames before laser activation was " + str(avgTempBefore))
                
                #Average Temp During Laser Activation
                avgTempDuring = 0
                for i in range (tempIndex, index + 1):
                    avgTempDuring += self.avgTempBox[i]
                    
                    #Accumulate pixel values for each frame
                    accumulated_pixels_during += self.img[:, :, i + framesCutOff]

                
                totalTempDuring += avgTempDuring
                totalFramesDuring += index - tempIndex + 1
                
                avgTempDuring /= index - tempIndex + 1
                self.avg_temp_during_list.append(avgTempDuring)
                
                #print("The average temperature during laser activation was " + str(avgTempDuring))
                
                
                #Average Temp After Laser Activation
                avgTempAfter20 = 0
                avgTempAfter = 0
                for i in range (index + 1, index + 1 + framesAfter):
                    avgTempAfter += self.avgTempBox[i]
                    if (i < index + 1 + 20):
                        avgTempAfter20 += self.avgTempBox[i]
                    
                
                for i in range(index + 1, index + 21):
                    # Accumulate pixel values for each frame
                    accumulated_pixels_after += self.img[:, :, i + framesCutOff]
                
                totalTempAfter20 += avgTempAfter20
                avgTempAfter20 /= 20
                self.avg_temp_after_list.append(avgTempAfter20)
                
                #print("The average temperature for the 20 frames after laser activation was ")
                
                totalTempAfter += avgTempAfter
                avgTempAfter /= framesAfter
                #print("The average temperature for the 75 frames after laser activation was " + str(avgTempAfter))
                
                for i in range(tempIndex - framesBefore, index + 21):
                    maxTemp = max(self.maxTempBox[i], maxTemp)
                
                self.maxTempTrial.append(maxTemp)
                
                indexTrial += 1
                
                continue
            
            index = int(frame) - framesCutOff - 1
            
            tempIndex = index
            doneFrame = True
       
        print('\n' + "The average temperature for all the trials combined in the 20 frames before activation was " + str(totalTempBefore/(numberActivations * framesBefore)))
        print("The average temperature for all the trials combined in the frames during activation was " + str(totalTempDuring/totalFramesDuring))
        print("The average temperature for all the trials combined in the 20 frames after activation was " + str(totalTempAfter20 / (numberActivations * 20)))
        print("The average temperature for all the trials combined in the 75 frames after activation was " + str(totalTempAfter/((numberActivations * framesAfter))) + '\n' * 3) 
       
        # Calculate the average values
        self.avg_temp_before_avg = np.mean(self.avg_temp_before_list)
        self.avg_temp_during_avg = np.mean(self.avg_temp_during_list)
        self.avg_temp_after_avg = np.mean(self.avg_temp_after_list)
        
        #Append Data of Each Plot for each Trial for framesBefore, during, and framesAfter
        startIndex = -1
        endIndex = -1

        i = 0
        
        #print("avgTempBox" + str(self.avgTempBox))
        print("lightOnFrames: " + str(self.lightOnFrames))
        
        for frame in self.lightOnFrames:
            #print("frame: " + str(frame))
            if (startIndex == -1):
                startIndex = int(frame) - framesCutOff
                continue
            
            endIndex = int(frame) - framesCutOff
            
            #self.all_plots_data.append(self.avgTempBox[startIndex-framesBefore : endIndex + 1 + framesAfter] / self.maxTempTrial[i])
            self.all_plots_data.append(self.avgTempBox[startIndex-framesBefore : endIndex + 1 + framesAfter])
            
            
            #print("startIndex-framesBefore: " + str(startIndex-framesBefore))
            #print("endIndex + 1 + framesAfter: " + str(endIndex + 1 + framesAfter))
            #print("all_plots_data " + str(i))
            #print(self.avgTempBox[startIndex-framesBefore : endIndex + 1 + framesAfter])
            
            i += 1
            startIndex = -1
        
        #print(self.maxTempTrial)
        
        #Set Thermal Image Averages
        self.average_pixels_before = accumulated_pixels_before / (framesBefore * numberActivations)
        self.average_pixels_during = accumulated_pixels_during / (totalFramesDuring)
        self.average_pixels_after = accumulated_pixels_after / (numberActivations * framesAfter)
    
    
    def plotAverageTempBox(self):
        #print("avgTempBox: " + str(self.avgTempBox))
         #print("lightOnFrames" + str(self.lightOnFrames))
        
        #Plot Average Temperature for 11x11 Box
        x = np.arange(len(self.avgTempBox))
        y = self.avgTempBox
        
        # Apply Savitzky-Golay filter for smoothing
        window_length = 11  # Must be an odd integer
        poly_order = 1  # Polynomial order of the filter
        smoothed_y = savgol_filter(y, window_length, poly_order)
        
        # Plot the original data and the smoothed data
        plt.plot(x, y, label='Smoothed Data')
        #plt.plot(x, smoothed_y, label='Smoothed Data', color='red')
        plt.xlabel('Frame Number (≈ seconds)')
        plt.ylabel('Average Temperature (°C)')
        plt.title('Average Temperature over Time for 11x11 Box of Pixels')
        plt.legend()
        
        '''plt.plot(self.avgTempBox)
        plt.xlabel('Frame Number (≈ seconds)')
        plt.ylabel('Average Temperature (°C)')
        plt.title('Average Temperature over Time for 11x11 Box of Pixels at Head')'''
        
        doneFrame = False
        tempIndex = 0
        y_line = max(self.avgTempBox)

        
        for frame in self.lightOnFrames:
            if (doneFrame):
                doneFrame = False
                index = int(frame) - framesCutOff - 1
                
                bottom = min(self.avgTempBox) - 0.1
                length = max(self.avgTempBox) - min(self.avgTempBox) + 0.2
                
                # Create the red rectangle
                rect = patches.Rectangle((tempIndex, bottom), index - tempIndex, length, edgecolor='red', facecolor='none')
                
                # Add the rectangle to the plot
                plt.gca().add_patch(rect)
                
                continue
            
            index = int(frame) - framesCutOff - 1
            tempIndex = index
            doneFrame = True
        
        plt.show()
    
    
    def plotAverageTempOverall(self):
        #Plot Average Temp for Whole Mouse
        plt.plot(self.averageTemp)
        plt.xlabel('Frame Number (≈ seconds)')
        plt.ylabel('Average Temperature (°C)')
        plt.title('Average Temperature over Time')

            
        doneFrame = False
        tempIndex = 0
        y_line = max(self.averageTemp) - 0.4

        for frame in self.lightOnFrames:
            if (doneFrame):
                doneFrame = False
                index = int(frame) - framesCutOff - 1
                
                # Create the red rectangle
                rect = patches.Rectangle((tempIndex, y_line - 0.65), index - tempIndex, 1.2, edgecolor='red', facecolor='none')
                
                # Add the rectangle to the plot
                plt.gca().add_patch(rect)
                
                #plt.plot([tempIndex, index], [y_line, y_line], color='blue', linestyle='-', linewidth=2)
                continue
            
            index = int(frame) - framesCutOff - 1
            tempIndex = index
            #plt.scatter(index, averageTemp[index], color='red', marker='o')
            doneFrame = True
        plt.show()    
    
    
    def plotMaxTemp(self):
        # Plot the maximum temperature over time
        plt.plot(self.maxTemp)
        plt.xlabel('Frame Number (≈ seconds)')
        plt.ylabel('Maximum Temperature (°C)')
        plt.title('Maximum Temperature over Time')

        doneFrame = False
        tempIndex = 0
        y_line = max(self.maxTemp) - 0.4

        for frame in self.lightOnFrames:
            if (doneFrame):
                doneFrame = False
                index = int(frame) - framesCutOff - 1
                
                # Create the red rectangle
                rect = patches.Rectangle((tempIndex, y_line - 0.65), index - tempIndex, 1.2, edgecolor='red', facecolor='none')
                
                # Add the rectangle to the plot
                plt.gca().add_patch(rect)
                
                #plt.plot([tempIndex, index], [y_line, y_line], color='blue', linestyle='-', linewidth=2)
                continue
            
            index = int(frame) - framesCutOff - 1
            tempIndex = index
            #plt.scatter(index, maxTemp[index], color='red', marker='o')
            doneFrame = True

        plt.show()
    
    
    def plotEachTrial(self):
        #Create Plots for Each Section

        startIndex = -1
        endIndex = -1

        for frame in self.lightOnFrames:
            if (startIndex == -1):
                startIndex = int(frame) - framesCutOff
                continue
            
            endIndex = int(frame) - framesCutOff
                
            x_values = range(startIndex - framesBefore, endIndex + 1 + framesAfter)
            
            window_length = 13  # Must be an odd integer
            poly_order = 1  # Polynomial order of the filter
            
            y = self.avgTempBox[startIndex-framesBefore : endIndex + 1 + framesAfter]
            smoothed_y = savgol_filter(y, window_length, poly_order)
            
            plt.plot(x_values, y)
            
            plt.axvline(x=startIndex, color='red', linestyle='dashed', alpha=0.5)
            plt.axvline(x=endIndex, color='red', linestyle='dashed', alpha=0.5)
            
            plt.xlabel('Frame Number (≈ seconds)')
            plt.ylabel(labelY)
            plt.title('Average Temperature over Time for 11x11 Box of Pixels at Head: One Trial')
            plt.show()
            
            startIndex = -1
    
    
    def plotAverageofTrials(self):
        # Find the maximum length among the plots
        max_length = max(len(plot) for plot in self.all_plots_data)
        min_length = min(len(plot) for plot in self.all_plots_data)
        #print("max_legnth: " + str(max_length))
        #print("min_length: " + str(min_length))
        #print("all plots data")
        #print(self.all_plots_data)
    
        # Pad shorter plots with NaN values to make them equal in length
        for i, plot in enumerate(self.all_plots_data):
            if len(plot) < max_length:
                padding_length = max_length - len(plot)
                self.all_plots_data[i] = np.pad(plot, (0, padding_length), 'constant', constant_values=np.nan)


        # Calculate the average of the data for all plots
        average_data = np.mean(self.all_plots_data, axis=0)
        #print("average_data" + str(average_data))
        
        sem = np.nanstd(self.all_plots_data, axis=0) / np.sqrt(len(self.all_plots_data))
        
        #print("Mouse is: " + self.name + "     All_Plots_Data is:  " + str(self.all_plots_data))

        #print("Mouse is: " + self.name + "     SEM is:  " + str(sem))
        
        # Plot the average data
        plt.plot(average_data, label = 'Average Temperature')
        plt.fill_between(range(len(average_data)), average_data - sem, average_data + sem, alpha=0.2, label='SEM', color='gray')
        plt.xlabel('Frame Number (≈ seconds)')
        plt.ylabel(labelY)
        plt.title('Average Temperature over Time for 11x11 Box of Pixels at Head\n(for ' +  str(framesBefore) + " frames before, during, and " + str(framesAfter) + ' frames after activation)')

        #Add Red Lines
        plt.axvline(x=framesBefore, color='red', linestyle='dashed', alpha=0.5)
        plt.axvline(x=max_length - framesAfter, color='red', linestyle='dashed', alpha=0.5)
            
        plt.legend()
        
        name = self.outputPath + "avgTrialGraph.svg"
        plt.savefig(name, format='svg')
        
        plt.show()
        
    
    def plotAverageofTrialsRat(self):
        # Find the maximum length among the plots
        min_length = min(len(plot) for plot in self.all_plots_data)
        #print("max_legnth: " + str(max_length))
        print("min_length: " + str(min_length))
        #print("all plots data")
        #print(self.all_plots_data)
        
        for i in range(len(self.all_plots_data)):
            self.all_plots_data[i] = self.all_plots_data[i][:min_length]

        # Calculate the average of the data for all plots
        average_data = np.mean(self.all_plots_data, axis=0)
        #print("average_data" + str(average_data))
        sem = np.nanstd(self.all_plots_data, axis=0) / np.sqrt(len(self.all_plots_data))
        
        # Plot the average data
        plt.plot(average_data, label = 'Average Temperature')
        plt.fill_between(range(len(average_data)), average_data - sem, average_data + sem, alpha=0.2, label='SEM', color='gray')
        plt.xlabel('Frame Number (≈ seconds)')
        plt.ylabel(labelY)
        plt.title('Average Temperature over Time for 11x11 Box of Pixels at Head\n(for ' +  str(framesBefore) + " frames before, during, and " + str(framesAfter) + ' frames after activation)')

        #Add Red Lines
        plt.axvline(x=framesBefore, color='red', linestyle='dashed', alpha=0.5)
            
        plt.legend()
        

        plt.show()
        
    
    
    def plotLineChartavgTempBeforeDuringAfter(self):
        #print("avgTempBeforeList" + str(self.avg_temp_before_list))
        #print("avgTempDuringList" + str(self.avg_temp_during_list))
        
        
        # Plot the average temperature for each trial separately
        plt.plot(self.avg_temp_before_list, label='Before')
        plt.plot(self.avg_temp_during_list, label='During')
        plt.plot(self.avg_temp_after_list, label='After')

        plt.axhline(self.avg_temp_before_avg, color='blue', linestyle='dashed', label=f'Before Avg ({self.avg_temp_before_avg:.2f})')
        plt.axhline(self.avg_temp_during_avg, color='orange', linestyle='dashed', label=f'During Avg ({self.avg_temp_during_avg:.2f})')
        plt.axhline(self.avg_temp_after_avg, color='green', linestyle='dashed', label=f'After Avg ({self.avg_temp_after_avg:.2f})')

        plt.xlabel('Trial Number')
        plt.ylabel('Average Temperature (°C)')
        plt.title('Average Temperature for 20 frames before, during, and after activation')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
      
   
    def averageThermalImages(self):
        #Thermal Images

        #[98:109, 249:260]
        #[70:150, 205:275]

        #Before Thermal Image

        print("Average Pixels Before")
        print(self.average_pixels_before)
    
        # Create a thermal image using a colormap
        plt.imshow(self.average_pixels_before[(self.center_col - 5):(self.center_col + 6), (self.center_row - 5):(self.center_row + 6)], cmap='hot', vmin=lowerLimit, vmax=upperLimit)
        plt.colorbar(label='Average Temperature (°C)')
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.title('Thermal Image of Average Temperatures in 20 Frames Before Activation')
        plt.show()


        #During Thermal Image

        # Create a thermal image using a colormap
        plt.imshow(self.average_pixels_during[(self.center_col - 5):(self.center_col + 6), (self.center_row - 5):(self.center_row + 6)], cmap='hot', vmin=lowerLimit, vmax=upperLimit)
        plt.colorbar(label='Average Temperature (°C)')
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.title('Thermal Image of Average Temperatures in Frames During Activation')
        plt.show()


        #After Thermal Image

        # Create a thermal image using a colormap
        plt.imshow(self.average_pixels_after[(self.center_col - 5):(self.center_col + 6), (self.center_row - 5):(self.center_row + 6)], cmap='hot', vmin=lowerLimit, vmax=upperLimit)
        plt.colorbar(label='Average Temperature (°C)')
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.title('Thermal Image of Average Temperatures in Frames After Activation')
        plt.show()
    
    
    def plotPupilSize(self):
        if (self.name == "mouse121"):
            xi = 570
            xf = 700
            yi = 237
            yf = 310
        
        elif (self.name == "mouse122"):
            xi = 615
            xf = 730
            yi = 170
            yf = 270
        
        elif (self.name == "mouse127"):
            xi = 560
            xf = 695
            yi = 150
            yf = 265
        elif (self.name == "mouse177"):
            xi = 560
            xf = 695
            yi = 150
            yf = 265
        elif (self.name == "mouse38"):
            xi = 520
            xf = 655
            yi = 110
            yf = 225
        elif (self.name == "mouse39"):
            xi = 560
            xf = 695
            yi = 150
            yf = 265
        elif (self.name == "mouse48"):
            xi = 560
            xf = 695
            yi = 150
            yf = 265
        else:
            xi = 560
            xf = 695
            yi = 150
            yf = 265
        
        saturation_factor = 0.1
        
        # Load video
        video_path = self.head_video_path
        cap = cv2.VideoCapture(video_path)
        
        # Check if video file is opened successfully
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()
        
        startIndex = -1
        endIndex = -1

        totalPupilArea = np.empty(0)        
        stdPupilArea = np.empty(0)

        #Individual Trial Plots
        for frameIndex in self.cameraFrameNumbers:
            if (startIndex == -1):
                startIndex = int(frameIndex)
                continue
            
            endIndex = startIndex + 605
                
            x_values = range(startIndex - 900, endIndex + 901)
            x_values = [x / 30 for x in x_values]
            pupil_areas_graph = self.pupil_areas[startIndex - 900:endIndex + 901]
            
            if (startIndex < 2000):
                totalPupilArea = pupil_areas_graph
                stdPupilArea = np.square(pupil_areas_graph)
                #print(startIndex)
                #print(type(totalPupilArea))
            else:
                #totalPupilArea = [x + y for x, y in zip(totalPupilArea, pupil_areas_graph)]
                
                totalPupilArea += np.array(pupil_areas_graph)
                stdPupilArea += np.square(pupil_areas_graph)
            
            #print(type(pupil_areas_graph))
            smooth_pupil_areas, _ = smooth(pupil_areas_graph)
            
            #plt.axvline(x=startIndex, color='red', linestyle='dashed', alpha=0.5)
            #plt.axvline(x=endIndex, color='red', linestyle='dashed', alpha=0.5)
            
            
            #print("Frame: " + str(frame))
            #print("Frame Numbers: " + str(self.frameNumbers))
            #print("Start Index: " + str(startIndex))
            #print("End Index: " + str(endIndex))
            #print("x-values: " + str(x_values))
            #print("Smooth pupil areas: " + str(smooth_pupil_areas))
            
            
            #Create a gridspec object with 1 row and 3 columns, and width ratios for each subplot
            fig = plt.figure(figsize=(12, 4))  # Adjust the figure size as needed
            gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
            
            # Plotting the first subplot
            ax1 = plt.subplot(gs[0])
            
            # Set the current frame position to the desired frame number
            cap.set(cv2.CAP_PROP_POS_FRAMES, startIndex)
            
            # Read the specific frame
            ret, frame = cap.read()
            
            if not ret:
                break
            
            sat_value = 24;
            frame_sat = np.copy(frame)
            frame_sat[frame_sat>sat_value] = sat_value
            frame_sat = np.int16(np.round(frame_sat * 255./sat_value));
            
            ax1.imshow(frame_sat)
            ax1.set_xlim(xi, xf)  # Set x-axis limits
            ax1.set_ylim(yi, yf)  # Set y-axis limits
            ax1.set_title(f"Before Activation")
            ax1.axis('off')
            
            # Set the current frame position to the desired frame number
            cap.set(cv2.CAP_PROP_POS_FRAMES, startIndex + 454)
            
            # Read the specific frame
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_sat = np.copy(frame)
            frame_sat[frame_sat>sat_value] = sat_value
            frame_sat = np.int16(np.round(frame_sat * 255./sat_value));
            
            ax3 = plt.subplot(gs[2])
            ax3.imshow(frame_sat)
            ax3.set_xlim(xi, xf)  # Set x-axis limits
            ax3.set_ylim(yi, yf)  # Set y-axis limits
            ax3.set_title(f"15s of Activation")
            ax3.axis('off')
            
            
            ax2 = plt.subplot(gs[1])
            startIndex /= 30
            endIndex /= 30
            
            ###Edits
            #
            #
            smooth_pupil_areas = smooth_pupil_areas[:2390]
            x_values = x_values[:2390]
            #
            #
            #
            
            ax2.axvline(x=startIndex, color='red', linestyle='dashed', alpha=0.5)
            ax2.axvline(x=endIndex, color='red', linestyle='dashed', alpha=0.5)
            laser_activation_patch = ax2.axvspan(startIndex, endIndex, color=(1.0, 0.8, 0.8), alpha=0.5, label = "Laser Activation Period")
            ax2.plot(x_values, smooth_pupil_areas, label='Pupil Area')
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Pupil Area (pixels^2)')
            ax2.set_title('Pupil Area Over Time')
            
            plt.tight_layout()
            plt.show()
            
            
            #Make Video
            #trialChosen = 12
            trialChosen = 1
            if (startIndex * 30 == self.cameraFrameNumbers[trialChosen]):     #Video Code
                begin = self.cameraFrameNumbers[trialChosen] - 900
                x_axis = range(0, 605 + 1801)
                for frame_num in range(begin, begin + 605 + 1801):
                    fig = plt.figure(figsize=(12, 4))  # Adjust the figure size as needed
                    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
                    fig.subplots_adjust(left=0.01)
                    ax1 = plt.subplot(gs[0])
                    # Set the current frame position to the desired frame number
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    
                    # Read the specific frame
                    ret, frame = cap.read()
                    
                    if not ret:
                        break
                    
                    sat_value = 24;
                    frame_sat = np.copy(frame)
                    frame_sat[frame_sat>sat_value] = sat_value
                    frame_sat = np.int16(np.round(frame_sat * 255./sat_value));
                    #plt.xlim(570, 700)
                    #plt.ylim(237, 310)
                    #plt.imshow(frame_sat)                    
                    
                    ax1.imshow(frame_sat)
                    ax1.set_xlim(xi, xf)  # Set x-axis limits
                    ax1.set_ylim(yi, yf)  # Set y-axis limits
                    ax1.set_title(f"Frame #" + str(frame_num))
                    ax1.axis('off')
                    
                    amountFrames = int(frame_num - begin)
                    video_pupil_areas = smooth_pupil_areas[0:amountFrames]
                    x = x_axis[0:amountFrames]
                    x = [num / 30 for num in x]
                    
                    ax2 = plt.subplot(gs[1])
                    ax2.plot(x, video_pupil_areas, label='Pupil Area')
                    if (frame_num >= begin + 900):
                        ax2.axvline(x = 30, color='red', linestyle='dashed', alpha=0.5)
                    if (frame_num >= begin + 900 + 605):
                        ax2.axvline(x=50.17, color='red', linestyle='dashed', alpha=0.5)
                        laser_activation_patch = ax2.axvspan(30, 50.17, color=(1.0, 0.8, 0.8), alpha=0.5, label = "Laser Activation Period")
                    ax2.set_xlabel('Time (seconds)')
                    ax2.set_ylabel('Pupil Area (pixels^2)')
                    ax2.set_title('Pupil Area Over Time (Mouse 127: Trial 7)')
                    ax2.set_xlim(0, 80.2)
                    ax2.set_ylim(400, 1950)
                    
                    name = self.outputPath + 'mouse127_Trial7_%06d.png' % (frame_num-begin)
                    #plt.savefig(name, dpi = 50)
                    
                    plt.show()
                          
            startIndex = -1
        
        #print("Total Pupil Area: " + str(totalPupilArea))
        
        totalPupilArea = np.array(totalPupilArea)
        #totalPupilArea = totalPupilArea[~np.isnan(totalPupilArea)]

        #x_values = range(self.cameraFrameNumbers[0] - 900, self.cameraFrameNumbers[1] + 901)
        x_values = range(0, 605 + 1801)
        x_values = [x / 30 for x in x_values]
        totalPupilArea = totalPupilArea/8
        self.avgPupilAreaAllTrials = totalPupilArea
        self.squarePupilAreasSum = stdPupilArea
        stdPupilArea = stdPupilArea/7
        stdPupilArea -= np.square(totalPupilArea) * 8 / 7
        stdPupilArea = np.sqrt(stdPupilArea) / np.sqrt(8)
        
        #print("Total Pupil Area")
        #print(totalPupilArea)
        
        #n = len(totalPupilArea)
        #mean_pupil_area = np.mean(totalPupilArea)
        #print("Mean Pupil Area is: " + str(mean_pupil_area))
        #std_deviation = np.std(totalPupilArea)
        #print("Standard Deviation is: " + str(std_deviation))
        #sem = std_deviation / np.sqrt(n)
        #print("Sem is: " + str(sem))
                
        smooth_pupil_areas, _ = smooth(totalPupilArea)
        
        laser_activation_patch = plt.axvspan(30, (self.cameraFrameNumbers[1] - self.cameraFrameNumbers[0])/30 + 30, color=(1.0, 0.8, 0.8), alpha=0.5, label = "Laser Activation Period")
        plt.axvline(x=30, color='red', linestyle='dashed', alpha=0.5)
        plt.axvline(x=(self.cameraFrameNumbers[1] - self.cameraFrameNumbers[0])/30 + 30, color='red', linestyle='dashed', alpha=0.5)
        
        #plt.axvline(x=self.cameraFrameNumbers[0]/30, color='red', linestyle='dashed', alpha=0.5)
        #plt.axvline(x=self.cameraFrameNumbers[1]/30, color='red', linestyle='dashed', alpha=0.5)
        
        ###Edits
        #
        #
        smooth_pupil_areas = smooth_pupil_areas[:2390]
        x_values = x_values[:2390]
        stdPupilArea = stdPupilArea[:2390]
        #
        #
        #
        
        plt.plot(x_values, smooth_pupil_areas, label='Pupil Area')
        plt.fill_between(x_values, smooth_pupil_areas - stdPupilArea, smooth_pupil_areas + stdPupilArea, alpha=0.5, label='SEM', color='gray')
        #plt.errorbar(x_values, smooth_pupil_areas, yerr=stdPupilArea, label='Pupil Area with SEM', color='b', ecolor='gray', capsize=5)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Pupil Area (Pixels^2)')
        plt.title('Average Pupil Area Over Time for Laser Activation')
        plt.legend()
        plt.tight_layout()
        plt.show()
                   
    
    def setUpSpeed(self):
        print("hi")
        
        
    def graphSpeed(self):
        print("hi")
    
    
    def saveDualGraph(self, trialNum):
        #for frame in range(lightOnFrames[8] - framesBefore, lightOnFrames[9] + framesAfter):
        #for frame in range(0, nframes - framesCutOff):
        

        
        # Data for the line graph
        #startLightFrame = 10
        #endLightFrame = 11
        startLightFrame = trialNum * 2 - 2
        endLightFrame = trialNum * 2 - 1
        
        print("Base Frame: " + str(self.frameNumbersActual[startLightFrame-2]))
        
        x_data = []
        y_data = []

        counter3 = 0
        red_line_counter = 0
        
        self.lightOnFrames = [x - framesCutOff for x in self.lightOnFrames]
        x = self.lightOnFrames[startLightFrame] - framesBefore
        y = self.lightOnFrames[endLightFrame] + framesAfter
        min_value = np.min(self.avgTempBox[x:y+1])
        max_value = np.max(self.avgTempBox[x:y+1])
        
        # Load video
        video_path = self.bodyPath
        cap = cv2.VideoCapture(video_path)
        
        # Check if video file is opened successfully
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()
        
        baseFrame = self.lightOnFrames[startLightFrame]
        #print("baseFrameNum: " + str(baseFrame))
        
        for frame in range(self.lightOnFrames[startLightFrame] - framesBefore, self.lightOnFrames[endLightFrame] + framesAfter):
            
            plt.figure(figsize=(12, 6))

            # Plot thermal image
            plt.subplot(1, 2, 1)
            
            center_row = 105
            center_col = 249
            
            row_start = max(center_row - 5, 0)
            row_end = min(center_row + 5, self.height)
            col_start = max(center_col - 5, 0)
            col_end = min(center_col + 5, self.width)
            
            # Make the Square
            square = patches.Rectangle((col_start, row_start), col_end - col_start, row_end - row_start, linewidth=0.5, edgecolor='blue', facecolor='none')
            
            #plt.imshow(self.img[70:150, 205:275, frame + framesCutOff], cmap='hot', vmin=lowerLimit, vmax=upperLimit)
            plt.gca().add_patch(square)
            
            #plt.imshow(self.img[98:109, 249:260, frame + framesCutOff], cmap='hot', vmin=lowerLimit, vmax=upperLimit)
            plt.imshow(self.img[0:239, 0:319, frame + framesCutOff], cmap='hot')
            plt.xlabel('Column')
            plt.ylabel('Row')
            plt.title('Thermal Image Frame ' + str(frame))
            plt.colorbar(label='Temperature (°C)')
            
            light_frame = 0
            
            if red_line_counter % 2 == 1:
                # Add text when the frame is within the range defined by the red lines
                plt.text(0.5, 0.95, 'Laser Activated', transform=plt.gca().transAxes,
                         ha='center', va='center', fontsize=16, fontweight='bold', color='red')
                    
            # Update the graph with one data point at a time
            x_data.append(frame)
            y_data.append(self.avgTempBox[frame])

            # Plot the graph
            plt.subplot(1, 2, 2)
            plt.plot(x_data, y_data)
            
            red_line_counter = 0
            for light_frame in self.lightOnFrames:
                if light_frame <= frame:
                    plt.axvline(x=light_frame, color='red', linestyle='dashed', alpha=0.5)
                    # Increment the red line counter
                    red_line_counter += 1
                    
            
            plt.xlabel('Frame Number (≈ seconds)')
            plt.ylabel('Average Temperature (°C)')
            plt.title('Average Temperature over Time for 11x11 Box of Pixels at Eye')
            
            # Manually set x and y axis limits
            #plt.xlim(-50, self.nframes - framesCutOff + 50)  # Set the x-axis limits
            plt.ylim(min_value - 0.04, max_value + 0.04)  # Set the y-axis limits
            plt.xlim(self.lightOnFrames[startLightFrame] - framesBefore - 10, self.lightOnFrames[endLightFrame] + framesAfter + 10)  # Set the x-axis limits
            
            
            
            #print("Body Frame: " + str((int)(((self.timeDifferencesPrefixSum[frame + framesCutOff] - self.timeDifferencesPrefixSum[baseFrame]))/1000*30)))
            #print("frame: " + str(frame + framesCutOff))
            #print("framestart: " + str(baseFrame))
            
            
            '''### Body Frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.frameNumbersActual[startLightFrame-2] + (int)(((self.timeDifferencesPrefixSum[frame] - self.timeDifferencesPrefixSum[baseFrame]))/1000*30))
            ret, frameBody = cap.read()
            plt.subplot(1, 3, 3)
            plt.imshow(frameBody)
            
            plt.tight_layout()'''
            
            #name = "image" + str(counter3) + ".png"
            name = self.outputPath + f'{self.imageName}_%05d.png' % counter3
            
            plt.savefig(name)
            
            counter3 += 1

            # Show the plot
            plt.show() 
        
        print("Done Saving Data")
           
   
    

#Combined Functions

def barChartNormal(mouse_objects):
   categories = ['Before', 'During', 'After']
   #categories = ['Before', 'After']
   width = 0.5
   x = np.arange(len(categories))
   yMin = 50
   yMax = 0
   legend_labels = []
   mouse_colors = ['blue', 'green', 'red', 'olive', 'orange', 'yellow']
   totalYBefore = []
   totalYDuring = []
   totalYAfter = []
   
   overall_avg_temp_before = np.mean([mouse.avg_temp_before_avg for mouse in mouse_objects])
   overall_avg_temp_during = np.mean([mouse.avg_temp_during_avg for mouse in mouse_objects])
   overall_avg_temp_after = np.mean([mouse.avg_temp_after_avg for mouse in mouse_objects])
   
   overall_averages = [overall_avg_temp_before, overall_avg_temp_during, overall_avg_temp_after]
   #overall_averages = [overall_avg_temp_before, overall_avg_temp_after]  
   
   # Set the size of the figure here (width, height)
   fig, ax = plt.subplots(figsize=(4, 5))
    
   #std_err_before = np.std([mouse.avg_temp_before_avg for mouse in mouse_objects], ddof=1) / np.sqrt(len(mouse_objects))
   #std_err_during = np.std([mouse.avg_temp_during_avg for mouse in mouse_objects], ddof=1) / np.sqrt(len(mouse_objects))
   #std_err_after = np.std([mouse.avg_temp_after_avg for mouse in mouse_objects], ddof=1) / np.sqrt(len(mouse_objects))
   
     
   # Plot individual dots for each trial's data points
   dot_size = 15
   x_noise_range = 0.15  # Adjust this value to control the amount of noise
   
   for i, category in enumerate(categories):
       for mouse_idx, mouse in enumerate(mouse_objects):
           if category == 'Before':
               x_values = [i - width / 2 + 0.25 + np.random.uniform(-x_noise_range, x_noise_range) for _ in mouse.avg_temp_before_list]
               y_values = [temp for temp in mouse.avg_temp_before_list]
           elif category == 'During':
               x_values = [i + np.random.uniform(-x_noise_range, x_noise_range) for _ in mouse.avg_temp_during_list]
               y_values = [temp for temp in mouse.avg_temp_during_list]
           elif category == 'After':
               x_values = [i + width / 2 - 0.25 + np.random.uniform(-x_noise_range, x_noise_range) for _ in mouse.avg_temp_after_list]
               y_values = [temp for temp in mouse.avg_temp_after_list]
           
           yMax = max(max(y_values), yMax)
           yMin = min(min(y_values), yMin)
           color = mouse_colors[mouse_idx % len(mouse_colors)]  # Get the color for the current Mouse object
           sc = ax.scatter(x_values, y_values, color=color, zorder=5, s=dot_size, label = mouse.name)  # Use the color for the scatter plot
           
           if (i == 0):
               legend_labels.append(sc) #'''
   
   '''for i, category in enumerate(categories):
        if category == 'Before':
            x_values = [i - width / 2 + 0.25 + np.random.uniform(-x_noise_range, x_noise_range) for mouse in mouse_objects for _ in mouse.avg_temp_before_list]
        elif category == 'During':
            x_values = [i + np.random.uniform(-x_noise_range, x_noise_range) for mouse in mouse_objects for _ in mouse.avg_temp_during_list]
        elif category == 'After':
            x_values = [i + width / 2 - 0.25 + np.random.uniform(-x_noise_range, x_noise_range) for mouse in mouse_objects for _ in mouse.avg_temp_after_list]
        
        if category == 'Before':
            y_values = [temp for mouse in mouse_objects for temp in mouse.avg_temp_before_list]
            totalYBefore += y_values
        if category == 'During':
            y_values = [temp for mouse in mouse_objects for temp in mouse.avg_temp_during_list]
            totalYDuring += y_values
        elif category == 'After':
            y_values = [temp for mouse in mouse_objects for temp in mouse.avg_temp_after_list]
            totalYAfter += y_values
        
        yMax = max(max(y_values), yMax)
        yMin = min(min(y_values), yMin)
        ax.scatter(x_values, y_values, color='black', zorder=5, s=dot_size)  # zorder ensures dots are plotted on top of bars'''
   

   std_err_before = sem(totalYBefore)
   std_err_during = sem(totalYDuring)
   std_err_after = sem(totalYAfter)

   ax.bar(x, overall_averages, width, label='Overall', color=['lightblue', 'lightgreen', 'lightcoral'], edgecolor='black', yerr=[std_err_before, std_err_during, std_err_after], capsize=5)
   #ax.bar(x, overall_averages, width, label='Overall', color=['lightblue', 'lightgreen'], edgecolor='black', yerr=[std_err_before, std_err_after], capsize=5)      
   
    
   ax.set_xlabel('Categories')
   ax.set_ylabel('Average Temperature (°C)')
   ax.set_title('Average Temperature for Each Category')
   ax.set_xticks(x)
   ax.set_xticklabels(categories)
   #ax.legend()
     
   #ax.set_ylim(31.2, 31.5)  # Set y-axis limits to 30.75°C to 32.25°C
   #ax.set_ylim(30.85, 31.8) #31.15
   ax.legend(handles=legend_labels, loc='upper left', bbox_to_anchor=(1, 1))
   ax.set_ylim(yMin - 0.02, yMax + 0.02)

   plt.tight_layout()
   plt.show()

def barChartDifference(mouse_objects):
    sumData = 0
    for mouse in mouse_objects:
        numValues = len(mouse.avg_temp_before_list) + len(mouse.avg_temp_during_list) + len(mouse.avg_temp_during_list)
        for value in mouse.avg_temp_before_list:
            sumData += value
        for value in mouse.avg_temp_during_list:
            sumData += value
        for value in mouse.avg_temp_during_list:
            sumData += value
        
        mouse.avg_temp_before_during_after = sumData / numValues
        sumData = 0
    
    mouse_colors = ['blue', 'green', 'red', 'olive', 'orange', 'yellow']
    categories = ['Before', 'During', 'After']
    #categories = ['Before', 'After']
    
    width = 0.5
    x = np.arange(len(categories))
    legend_labels = []
    totalYBefore = []
    totalYDuring = []
    totalYAfter = []
    
    overall_avg_temp_before = np.mean([mouse.avg_temp_before_avg - mouse.avg_temp_before_during_after for mouse in mouse_objects])
    overall_avg_temp_during = np.mean([mouse.avg_temp_during_avg - mouse.avg_temp_before_during_after for mouse in mouse_objects])
    overall_avg_temp_after = np.mean([mouse.avg_temp_after_avg - mouse.avg_temp_before_during_after for mouse in mouse_objects])
    
    overall_averages = [overall_avg_temp_before, overall_avg_temp_during, overall_avg_temp_after]
    #overall_averages = [overall_avg_temp_before, overall_avg_temp_after]  
    
    # Set the size of the figure here (width, height)
    fig, ax = plt.subplots(figsize=(4, 5))
      
    #std_err_before = np.std([mouse.avg_temp_before_avg for mouse in mouse_objects], ddof=1) / np.sqrt(len(mouse_objects))
    #std_err_during = np.std([mouse.avg_temp_during_avg for mouse in mouse_objects], ddof=1) / np.sqrt(len(mouse_objects))
    #std_err_after = np.std([mouse.avg_temp_after_avg for mouse in mouse_objects], ddof=1) / np.sqrt(len(mouse_objects))
      
    # Plot individual dots for each trial's data points
    
    yMin = 50
    yMax = 0
    
    dot_size = 15
    x_noise_range = 0.15  # Adjust this value to control the amount of noise
    '''for i, category in enumerate(categories):
        if category == 'Before':
             x_values = [i - width / 2 + 0.25 + np.random.uniform(-x_noise_range, x_noise_range) for mouse in mouse_objects for _ in mouse.avg_temp_before_list]
        elif category == 'During':
             x_values = [i + np.random.uniform(-x_noise_range, x_noise_range) for mouse in mouse_objects for _ in mouse.avg_temp_during_list]
        elif category == 'After':
             x_values = [i + width / 2 - 0.25 + np.random.uniform(-x_noise_range, x_noise_range) for mouse in mouse_objects for _ in mouse.avg_temp_after_list]
     
        y_values = [temp - mouse.avg_temp_before_during_after for mouse in mouse_objects for temp in mouse.avg_temp_before_list]
        if category == 'During':
             y_values = [temp - mouse.avg_temp_before_during_after for mouse in mouse_objects for temp in mouse.avg_temp_during_list]
        elif category == 'After':
             y_values = [temp - mouse.avg_temp_before_during_after for mouse in mouse_objects for temp in mouse.avg_temp_after_list]
        
        color = mouse_colors[i % len(mouse_colors)] 
        ax.scatter(x_values, y_values, color='black', zorder=5, s=dot_size)  # zorder ensures dots are plotted on top of bars'''
    
    for i, category in enumerate(categories):
        for mouse_idx, mouse in enumerate(mouse_objects):
            if category == 'Before':
                x_values = [i - width / 2 + 0.25 + np.random.uniform(-x_noise_range, x_noise_range) for _ in mouse.avg_temp_before_list]
                y_values = [temp - mouse.avg_temp_before_during_after for temp in mouse.avg_temp_before_list]
                totalYBefore += y_values
            elif category == 'During':
                x_values = [i + np.random.uniform(-x_noise_range, x_noise_range) for _ in mouse.avg_temp_during_list]
                y_values = [temp - mouse.avg_temp_before_during_after for temp in mouse.avg_temp_during_list]
                totalYDuring += y_values
            elif category == 'After':
                x_values = [i + width / 2 - 0.25 + np.random.uniform(-x_noise_range, x_noise_range) for _ in mouse.avg_temp_after_list]
                y_values = [temp - mouse.avg_temp_before_during_after for temp in mouse.avg_temp_after_list]
                totalYAfter += y_values
            
            yMax = max(max(y_values), yMax)
            yMin = min(min(y_values), yMin)
            color = mouse_colors[mouse_idx % len(mouse_colors)]  # Get the color for the current Mouse object
            sc = ax.scatter(x_values, y_values, color=color, zorder=5, s=dot_size, label = mouse.name)  # Use the color for the scatter plot
            
            if (i == 0):
                legend_labels.append(sc)

    
    std_err_before = sem(totalYBefore)
    std_err_during = sem(totalYDuring)
    std_err_after = sem(totalYAfter)
    
    ax.bar(x, overall_averages, width, label='Overall', color=['lightblue', 'lightgreen', 'lightcoral'], edgecolor='black', yerr=[std_err_before, std_err_during, std_err_after], capsize=5, zorder = 5, error_kw=dict(ecolor='black', lw=2))
    #ax.bar(x, overall_averages, width, label='Overall', color=['lightblue', 'lightgreen'], edgecolor='black', yerr=[std_err_before, std_err_after], capsize=5, zorder = 5, error_kw=dict(ecolor='black', lw=2))
    
    plt.axhline(y = 0, color = 'lightgray', linestyle = 'solid')  
    
    ax.set_ylabel('Average Temperature Difference from Mean (°C)')
    ax.set_title('Average Temperature Difference from Mean for Each Category')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    #ax.legend()
    ax.legend(handles=legend_labels, loc='upper left', bbox_to_anchor=(1, 1))
 
    #ax.set_ylim(-0.4, 0.45)
    ax.set_ylim(yMin - 0.02, yMax + 0.02)

    plt.tight_layout()
    plt.show()
    
def avgTrialLineGraph(mouse_objects):
    all_mouse_data = []
    
    for mouse in mouse_objects:
        for plot in mouse.all_plots_data:
            all_mouse_data.append(plot)
    
    # Find the maximum length among the plots
    max_length = max(len(plot) for plot in all_mouse_data)

    # Pad shorter plots with NaN values to make them equal in length
    for i, plot in enumerate(all_mouse_data):
        if len(plot) < max_length:
            padding_length = max_length - len(plot)
            all_mouse_data[i] = np.pad(plot, (0, padding_length), 'constant', constant_values=np.nan)


    # Calculate the average of the data for all plots
    average_data = np.mean(all_mouse_data, axis=0)
    sem = np.nanstd(all_mouse_data, axis=0) / np.sqrt(len(all_mouse_data))

    # Plot the average data
    plt.plot(average_data, label = 'Average Temperature')
    plt.fill_between(range(len(average_data)), average_data - sem, average_data + sem, alpha=0.2, label='SEM', color='gray')
    plt.xlabel('Time (Seconds)', fontsize = 14)
    plt.ylabel("Average Temperature (°C)", fontsize = 14)
    #plt.title('Average Temperature over Time for 11x11 Box of Pixels\n(At Head for ' +  str(framesBefore) + " frames before, during, and " + str(framesAfter) + ' frames after activation)')
    plt.title('Average Temperature over Time at Box for Laser Activation', fontsize = 16)
    
    
    #Add Red Lines
    laser_activation_patch = plt.axvspan(framesBefore, max_length - framesAfter, color=(1.0, 0.8, 0.8), alpha=0.5, label = "Laser Activation Period")
    plt.axvline(x=framesBefore, color='red', linestyle='dashed', alpha=0.5)
    plt.axvline(x=max_length - framesAfter, color='red', linestyle='dashed', alpha=0.5)
    
    #plt.text((framesBefore + max_length - framesAfter) / 2, 32, 'Laser Activation Period', color='red', fontsize=12, ha='center')
    # Add label for the shaded area in the legend
    #plt.legend(handles=[laser_activation_patch], labels=['Laser Activation Period'])
    
    x_values = np.arange(framesBefore, max_length - framesAfter)
    y_values = average_data[framesBefore:max_length - framesAfter]
    slope, intercept = np.polyfit(x_values, y_values, 1)
    
    # Create linear regression line equation
    regression_line = slope * x_values + intercept

    # Plot the linear regression line
    #plt.plot(x_values, regression_line, color='black', label=f'Linear Regression\n(y = {slope:.6f}x + {intercept:.3f})')
    n1 = mouse_objects[0].name
    
    name = "/Users/david/Documents/Programming_Projects/CaltechProject/VideoImages/savePlots/" + n1 + "_avgTrialLineGraph.svg"
    plt.savefig(name, format='svg')
    
    plt.legend()
    plt.show()
    

def avgTrialLineGraphRat(mouse_objects):
    all_mouse_data = []
    
    for mouse in mouse_objects:
        for plot in mouse.all_plots_data:
            all_mouse_data.append(plot)
    
    # Find the maximum length among the plots
    min_length = min(len(plot) for plot in all_mouse_data)

    for i in range(len(all_mouse_data)):
        all_mouse_data[i] = all_mouse_data[i][:min_length]

    # Calculate the average of the data for all plots
    average_data = np.mean(all_mouse_data, axis=0)
    sem = np.nanstd(all_mouse_data, axis=0) / np.sqrt(len(all_mouse_data))

    # Plot the average data
    plt.plot(average_data, label = 'Average Temperature')
    plt.fill_between(range(len(average_data)), average_data - sem, average_data + sem, alpha=0.2, label='SEM', color='gray')
    plt.xlabel('Time (Seconds)', fontsize = 14)
    plt.ylabel("Average Temperature (°C)", fontsize = 14)
    #plt.title('Average Temperature over Time for 11x11 Box of Pixels\n(At Head for ' +  str(framesBefore) + " frames before, during, and " + str(framesAfter) + ' frames after activation)')
    plt.title(f'Average Temperature over Time at Box for Laser Activation: Rat Stim, Far, {len(all_mouse_data)} trials', fontsize = 16)
    
    
    #Add Red Lines
    plt.axvline(x=framesBefore, color='red', linestyle='dashed', alpha=0.5)
    
    
    #x_values = np.arange(framesBefore, min_length)
    #y_values = average_data[framesBefore:max_length - framesAfter]
    #slope, intercept = np.polyfit(x_values, y_values, 1)
    
    # Create linear regression line equation
    #regression_line = slope * x_values + intercept

    # Plot the linear regression line
    #plt.plot(x_values, regression_line, color='black', label=f'Linear Regression\n(y = {slope:.6f}x + {intercept:.3f})')
    
    n1 = mouse_objects[0].name
    name = "/Users/david/Documents/Programming_Projects/CaltechProject/VideoImages/savePlots/" + n1 + "_avgTrialLineGraphRat.svg"
    plt.savefig(name, format='svg')
    
    plt.legend()
    plt.show()

def normalizedBarChart(mouse_objects):
    categories = ['Before', 'After']
    width = 0.5
    x = np.arange(len(categories))
    legend_labels = []
    mouse_colors = ['blue', 'green', 'red', 'olive', 'orange', 'yellow']
    totalYBefore = []
    totalYDuring = []
    totalYAfter = []
    yMin = 2
    yMax = 0
    
    overall_avg_temp_before = np.mean([[mouse.avg_temp_before_list[i] / mouse.maxTempTrial[i] for i in range(len(mouse.avg_temp_before_list))] for mouse in mouse_objects])
    overall_avg_temp_during = np.mean([[mouse.avg_temp_during_list[i] / mouse.maxTempTrial[i] for i in range(len(mouse.avg_temp_during_list))] for mouse in mouse_objects])
    overall_avg_temp_after = np.mean([[mouse.avg_temp_after_list[i] / mouse.maxTempTrial[i] for i in range(len(mouse.avg_temp_after_list))] for mouse in mouse_objects])
    
    overall_averages = [overall_avg_temp_before, overall_avg_temp_after]
      
    # Set the size of the figure here (width, height)
    fig, ax = plt.subplots(figsize=(4, 5))
     
    #std_err_before = np.std([mouse.avg_temp_before_avg for mouse in mouse_objects], ddof=1) / np.sqrt(len(mouse_objects))
    #std_err_during = np.std([mouse.avg_temp_during_avg for mouse in mouse_objects], ddof=1) / np.sqrt(len(mouse_objects))
    #std_err_after = np.std([mouse.avg_temp_after_avg for mouse in mouse_objects], ddof=1) / np.sqrt(len(mouse_objects))
    
      
    # Plot individual dots for each trial's data points
    dot_size = 15
    x_noise_range = 0.15  # Adjust this value to control the amount of noise
    
    for i, category in enumerate(categories):
        for mouse_idx, mouse in enumerate(mouse_objects):
            if category == 'Before':
                x_values = [i - width / 2 + 0.25 + np.random.uniform(-x_noise_range, x_noise_range) for _ in mouse.avg_temp_before_list]
                y_values = [mouse.avg_temp_before_list[i] / mouse.maxTempTrial[i] for i in range(len(mouse.avg_temp_before_list))]
                totalYBefore+=(y_values)
            elif category == 'During':
                x_values = [i + np.random.uniform(-x_noise_range, x_noise_range) for _ in mouse.avg_temp_during_list]
                y_values = [mouse.avg_temp_during_list[i] / mouse.maxTempTrial[i] for i in range(len(mouse.avg_temp_during_list))]
                totalYDuring+=(y_values)
            elif category == 'After':
                x_values = [i + width / 2 - 0.25 + np.random.uniform(-x_noise_range, x_noise_range) for _ in mouse.avg_temp_after_list]
                y_values = [mouse.avg_temp_after_list[i] / mouse.maxTempTrial[i] for i in range(len(mouse.avg_temp_after_list))]
                totalYAfter+=(y_values)
            
            color = mouse_colors[mouse_idx % len(mouse_colors)]  # Get the color for the current Mouse object
            sc = ax.scatter(x_values, y_values, color=color, zorder=5, s=dot_size, label = mouse.name)  # Use the color for the scatter plot
            
            yMax = max(max(y_values), yMax)
            yMin = min(min(y_values), yMin)
            
            if (i == 0):
                legend_labels.append(sc)    

    std_err_before = sem(totalYBefore)
    std_err_during = sem(totalYDuring)
    std_err_after = sem(totalYAfter)
    
    std_err = [std_err_before, std_err_after]
    
    ax.bar(x, overall_averages, width, label='Overall', color=['lightblue', 'lightgreen'], edgecolor='black', yerr=std_err, capsize=5)
          
    ax.set_xlabel('Categories')
    ax.set_ylabel('% of Max Temperature')
    ax.set_title('Normalized Average Temperature Before & After Actviation')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    #ax.legend()
      
    #ax.set_ylim(31.2, 31.5)  # Set y-axis limits to 30.75°C to 32.25°C
    #ax.set_ylim(30.85, 31.8) #31.15
    ax.legend(handles=legend_labels, loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylim(yMin - 0.002, yMax + 0.002)

    plt.tight_layout()
    plt.show()    

def tempAfterBar(mouse_objects):
    sumData = 0
    for mouse in mouse_objects:
        numValues = len(mouse.avg_temp_before_list) + len(mouse.avg_temp_during_list) + len(mouse.avg_temp_after_list)
        for value in mouse.avg_temp_before_list:
            sumData += value
        for value in mouse.avg_temp_during_list:
            sumData += value
        for value in mouse.avg_temp_after_list:
            sumData += value
        
        mouse.avg_temp_before_during_after = sumData / numValues
        sumData = 0
    
    mouse_colors = ['blue', 'green', 'red', 'olive', 'orange', 'yellow']
    categories = ["After - Before"]
    
    width = 0.5
    x = np.arange(len(categories))
    legend_labels = []
    totalYAfter = []
    
    overall_avg_temp_after = np.mean([mouse.avg_temp_after_avg - mouse.avg_temp_before_during_after for mouse in mouse_objects])
    
    #overall_averages = [overall_avg_temp_before, overall_avg_temp_during, overall_avg_temp_after]
    overall_averages = [overall_avg_temp_after]  
    
    # Set the size of the figure here (width, height)
    fig, ax = plt.subplots(figsize=(4, 5))
      
    # Plot individual dots for each trial's data points
    
    yMin = 50
    yMax = 0
    
    dot_size = 15
    x_noise_range = 0.15  # Adjust this value to control the amount of noise

    for mouse_idx, mouse in enumerate(mouse_objects):
        x_values = [width / 2 - 0.25 + np.random.uniform(-x_noise_range, x_noise_range) for _ in mouse.avg_temp_after_list]
        y_values =[after - before for after, before in zip(mouse.avg_temp_after_list, mouse.avg_temp_before_list)]
        totalYAfter += y_values
        
        yMax = max(max(y_values), yMax)
        yMin = min(min(y_values), yMin)
        color = mouse_colors[mouse_idx % len(mouse_colors)]  # Get the color for the current Mouse object
        sc = ax.scatter(x_values, y_values, color=color, zorder=6, s=dot_size, label = mouse.name)  # Use the color for the scatter plot
        
        legend_labels.append(sc)

    
    std_err_after = sem(totalYAfter)
    
    #ax.bar(x, overall_averages, width, label='Overall', color=['lightblue', 'lightgreen', 'lightcoral'], edgecolor='black', yerr=[std_err_before, std_err_during, std_err_after], capsize=5, zorder = 5, error_kw=dict(ecolor='black', lw=2))
    ax.bar(x, overall_averages, width, label='Overall', color=['lightblue'], edgecolor='black', yerr=[std_err_after], capsize=5, zorder = 5, error_kw=dict(ecolor='black', lw=2))
    
    plt.axhline(y = 0, color = 'lightgray', linestyle = 'solid')  
    
    #ax.set_xlabel('')
    ax.set_ylabel('Temperature Difference (°C)', fontsize = 14)
    ax.set_title('Average Temperature Difference for Laser Activations', fontsize = 16)
    ax.set_xticks(x)
    #ax.set_xticklabels(categories)
    #ax.legend()
    ax.legend(handles=legend_labels, loc='upper left', bbox_to_anchor=(1, 1))
 
    #ax.set_ylim(-0.4, 0.45)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(yMin - 0.02, yMax + 0.02)

    plt.tight_layout()
    plt.show()

def pairedPlot(mouse_objects):
    #Normal Data
    legend_labels = []
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    colors2 = ['blue', 'green', 'red']
    offset = 0.02
    totalBefore = []
    totalAfter = []
    
    for idx, mouse in enumerate(mouse_objects):
        before_list = mouse.avg_temp_before_list
        after_list = mouse.avg_temp_after_list
        
        totalBefore += before_list
        totalAfter += after_list
        
        #Normalize it
        #for i in range(numberActivations):
            #before_list[i] /= mouse.maxTempTrial[i]
            #after_list[i] /= mouse.maxTempTrial[i]
            
        
        beforeAvg = np.mean(before_list)
        afterAvg = np.mean(after_list)
        
        
        avg_temp_diff = (afterAvg - beforeAvg)
        
        # Make sure both lists have the same length
        assert len(before_list) == len(after_list), "The two lists must have the same length"
        
        line = plt.plot(['0 to 20s', '40 to 60'], [before_list, after_list], marker='o', linestyle='-', color = colors[idx], alpha = 0.65, linewidth = 1, label = f'{mouse.name}\n(Temperature Increase: {avg_temp_diff:.4f})')
        
        legend_labels.append(line[0])
        #plt.plot([0, 1], [beforeAvg, afterAvg], color=colors2[idx], linestyle='--', linewidth = 3, marker='o', zorder = 5)
        
    
    avgBefore = np.mean(totalBefore)
    avgAfter = np.mean(totalAfter)
    diff = avgAfter - avgBefore
    
    avgLine = plt.plot([0, 1], [avgBefore, avgAfter], color='black', linestyle='--', linewidth = 3.5, marker='o', zorder = 5, label = f'Average Temperature Increase\n(Temperature Increase: {diff:.4f})')
    #legend_labels.append(avgLine)
    
    # Set labels and title
    #plt.xlabel('Category')
    plt.ylabel(labelY)
    #plt.title('Paired Plot of Average Temperature Before vs. After')
    
    
    # Adjust the subplot parameters
    #plt.subplots_adjust(left=1, right=1.275)
    plt.subplots_adjust(left=0.1, right=0.4, top=0.9, bottom=0.1)
    
    #plt.xlim(0.15, 0.9)
    #plt.ylim(np.min(totalBefore) - 0.02, np.max(totalAfter) + 0.02)
    
    
    # Add the legend with specified number of labels
    # Add the legend with all labels
    plt.legend(handles=legend_labels, loc='upper left', bbox_to_anchor=(1, 1))
    #plt.ylim(np.min(totalBefore) - 0.02, np.max(totalAfter) + 0.02)
    difference = difference = [x - y for x, y in zip(totalAfter, totalBefore)]
    print("Performing Tests")
    print(stats.kruskal(totalBefore, totalAfter))
    print(stats.ttest_rel(totalBefore, totalAfter))
    res = wilcoxon(difference)
    print("Wilcoxon p-value: " + str(res.pvalue))
    
    p_value = res.pvalue
    if p_value < 0.001:
       significance_text = '***'
    elif p_value < 0.01:
       significance_text = '**'
    elif p_value < 0.05:
       significance_text = '*'
    else:
       significance_text = 'ns'
    
    #plt.text(1.5, 31.08, "Wilcoxon Test\n", fontsize=21, color='black', fontweight="bold", ha='center')
    #plt.text(1.5, 31, f'$p = {p_value:.5f}$\n{significance_text}', fontsize=15, color='black', ha='center')
    plt.text(1.9, avgBefore-0.1, "Wilcoxon Test\n", fontsize=21, color='black', fontweight="bold", ha='center')
    plt.text(1.9, avgBefore-0.25, f'$p = {p_value:.5f}$\n{significance_text}', fontsize=15, color='black', ha='center')
    
    
    
    #plt.text(2, 21.19, "Wilcoxon Test\n", fontsize=21, color='black', fontweight="bold", ha='center')
    #plt.text(2, 21.16, f'$p = {p_value:.5f}$\n{significance_text}', fontsize=15, color='black', ha='center')
    plt.draw()
    
    n1 = mouse_objects[0].name
    name = f"/Users/david/Documents/Programming_Projects/CaltechProject/VideoImages/savePlots/{n1}_pairedPlot.svg"
    plt.savefig(name, format='svg')
    
    # Show the plot
    plt.show()
    
def plotLinearRegression(mouse_objects):
    all_mouse_data = []
    predicted = []
    
    for mouse in mouse_objects:
        for plot in mouse.all_plots_data:
            all_mouse_data.append(plot)
    
    # Find the maximum length among the plots
    
    max_length = max(len(plot) for plot in all_mouse_data)

    # Pad shorter plots with NaN values to make them equal in length
    for i, plot in enumerate(all_mouse_data):
        if len(plot) < max_length:
            padding_length = max_length - len(plot)
            all_mouse_data[i] = np.pad(plot, (0, padding_length), 'constant', constant_values=np.nan)
    
    # Calculate the average of the data for all plots
    average_data = np.mean(all_mouse_data, axis=0)
    
    
    
    # Plot the average data
    plt.plot()
    plt.xlabel('Seconds')
    plt.ylabel(labelY)
    plt.title('Linear Regression During Activation')
    
    x_values = np.arange(framesBefore, max_length - framesAfter)
    y_values = average_data[framesBefore:max_length - framesAfter]
    slope, intercept = np.polyfit(x_values, y_values, 1)
    
    for i in range(framesBefore, max_length - framesAfter):
        predicted.append(slope * i + intercept)
    
    # Create linear regression line equation
    regression_line = slope * x_values + intercept
    
    # Plot the individual points using scatter plot
    plt.scatter(x_values, y_values, color='blue', marker='o', label='Individual Points')
    
    rsquared = r2_score(predicted, y_values)
    
    # Plot the linear regression line
    plt.plot(x_values, regression_line, color='black', label=f'Actual Linear Regression\n(y = {slope:.6f}x + {intercept:.3f})\nr^2: {rsquared:.4f}')
    
    slope2 = 0.001325
    intercept2 = intercept + slope * 20  - slope2 * 20
    regression_line2 = slope2 * x_values + intercept2
    plt.plot(x_values, regression_line2, color='red', linestyle='dashed', label=f'Predicted Linear Regression')
    
    n1 = mouse_objects[0].name
    name = "/Users/david/Documents/Programming_Projects/CaltechProject/VideoImages/savePlots/" + n1 + "_linearRegression.svg"
    plt.savefig(name, format='svg')
    
    plt.legend()
    plt.show()

def dostuff():
    filename = "/Users/david/Downloads/mouse122_1_hzcontBodyDLC_resnet50_Body Track MalesJul5shuffle1_100000_filtered.h5"
    dataset = "df_with_missing/table"
    window_length = 5  # Must be an odd integer
    poly_order = 1  # Polynomial order of the filter

    #Order Columns: left_back_hand, right_back_hand, left_forward_hand, right_forward_hand


    # Open the HDF5 file in read mode
    def print_dataset_content(dataset):
        print("Dataset:", dataset)
        data = file[dataset][:]
        print("Content:")
        print(data)
        print("\n")
        
    def plotGraph (dataset, x, name, xname = "X Value"):
        title = "Tracking Mouse Speed over Time"
        plt.plot(x, dataset)
        plt.xlabel('Time (seconds)')
        plt.ylabel(xname)
        plt.title(title)
        #plt.xlim(266, 316)
        plt.show()

    def smooth(x,window_len=60,window='hanning'):

        s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
        #print(len(s))
        if window == 'flat': #moving average
            w=np.ones(window_len,'d')
        else:
            w=eval('np.'+window+'(window_len)')

        y=np.convolve(w/w.sum(),s,mode='valid')
        return y


    with h5py.File(filename, 'r') as file:
        
        data = file[dataset][:]

    #Declaring Vars
    left_back_hand_x = [row_values[0] for _, row_values in data]
    right_back_hand_x = [row_values[3] for _, row_values in data]
    left_forward_hand_x = [row_values[6] for _, row_values in data]
    right_forward_hand_x = [row_values[9] for _, row_values in data]



    #plotGraph(left_back_hand_x, "Left Back Hand X Coordinate")
    #plotGraph(right_back_hand_x, "Right Back Hand X Coordinate")
    #plotGraph(left_forward_hand_x, "Left Forward Hand X Coordinate")
    #plotGraph(right_forward_hand_x, "Right Forward Hand X Coordinate")

    time_in_seconds = []

    speed = []    

    for i in range(1, len(left_back_hand_x)):
        time_per_frame = 1 / 30
        #average = ((left_back_hand_x[i] - left_back_hand_x[i-1]) + (right_back_hand_x[i] - right_back_hand_x[i-1]) + (left_forward_hand_x[i] - left_forward_hand_x[i-1]) + (right_forward_hand_x[i] - right_forward_hand_x[i-1]))/4
        average = abs((abs((left_back_hand_x[i] - left_back_hand_x[i-1])) + abs((right_back_hand_x[i] - right_back_hand_x[i-1])))/2)
        #average = (right_back_hand_x[i] - right_back_hand_x[i-1])
        #if (average < 0):
            #average = 0
        
        speed.append(average)
        time_in_seconds.append(i * time_per_frame)
        
        


    speed = smooth(speed)
    speed = speed[:36910]
    time_in_seconds = time_in_seconds[:36910]

    plotGraph(speed, time_in_seconds, "Average Difference in X Coordinate", "Pixels/s")
    
    return speed

def graphSpeed(speed, avgTempBox, max_display_frames=100):
    
    
    #Temperature 
    framesSlowRun = []
    totalTempSlowRun = 0
    totalFramesSlowRun = 0
    
    framesNoRun = []
    totalTempNoRun = 0
    totalFramesNoRun = 0
    
    
    #Graph Average Temp: Run vs No Run
    #
    #
    print("Speed before is " + str(len(speed)))
    
    frame = int(mouse122.timeDiffCutoff / 1000 * 30)
    speed = speed[frame:]
    
    print("Speed After is " + str(len(speed)))
    
    num_frames = len(avgTempBox) - 5
    totalTime = 0
    current_frame = 0
    onPreviousRoute = False
    previousRouteStart = -50
    totalTemperatureRun = 0
    totalTemperature = 0
    totalFramesRun = 0
    totalTime = []
    
    framesRun = []
    
    #print(mouse122.timeDifferences)
    
    while current_frame < num_frames:
        if (current_frame > 0):
            totalTime.append(totalTime[current_frame-1] + mouse122.timeDifferences[current_frame])
        else:
            totalTime.append(mouse122.timeDifferences[0])
        
        frameSpeed = int(totalTime[current_frame] / 1000 * 30)
        
        if (speed[frameSpeed] > 20 and onPreviousRoute == False):
            previousRouteStart = current_frame
            onPreviousRoute = True
    
        if (current_frame < previousRouteStart + 41 and onPreviousRoute == True):
            totalFramesRun += 1
            totalTemperatureRun += avgTempBox[current_frame]
            framesRun.append(avgTempBox[current_frame])
        
        if (current_frame == previousRouteStart + 41 and onPreviousRoute == True):
            onPreviousRoute = False
            
        if (speed[frameSpeed] > 1 and speed[frameSpeed] < 3):
            totalFramesSlowRun += 1
            totalTempSlowRun += avgTempBox[current_frame]
            framesSlowRun.append(avgTempBox[current_frame])
            
        if (speed[frameSpeed] < 1):
            totalFramesNoRun += 1
            totalTempNoRun += avgTempBox[current_frame]
            framesNoRun.append(avgTempBox[current_frame])
            
        totalTemperature += avgTempBox[current_frame]
        
        current_frame += 1
        
        
    totalTemperature /= (num_frames)
    totalTemperatureRun /= totalFramesRun
    avgTemperatureSlowrun = totalTempSlowRun / totalFramesSlowRun
    avgTempNoRun = totalTempNoRun / totalFramesNoRun
    
    sem_temperature = np.std(avgTempBox[:num_frames]) / np.sqrt(num_frames)
    sem_temperature_run = np.std(framesRun) / np.sqrt(totalFramesRun)
    sem_temperature_slowrun = np.std(framesSlowRun) / np.sqrt(totalFramesSlowRun)
    sem_temperature_norun = np.std(framesNoRun) / np.sqrt(totalFramesNoRun)
    
    # Names for the bars
    variables2 = ['Avg Temp', 'Avg Temp \n Run', "Avg Temp \n Slow-Run", "Avg Temp \n No-Run"]
    
    # Values for the bars
    values2 = [totalTemperature, totalTemperatureRun, avgTemperatureSlowrun, avgTempNoRun]
    
    # Create the bar chart
    plt.bar(variables2, values2, yerr=[sem_temperature, sem_temperature_run, sem_temperature_slowrun, sem_temperature_norun], color=['blue', 'green', "red", "orange"], capsize=5)
    
    # Add labels and title
    plt.ylabel('Temperature (˚C)')
    plt.ylim(min(totalTemperature, totalTemperatureRun) - 0.05, max(totalTemperature, totalTemperatureRun) + 0.05)
    plt.title('Bar Chart of Temp: Run vs No Run')
    
    # Display the plot
    plt.show()
    
    
    
    #Video 
    #
    #
    print("Timedifcutoff is " + str(mouse122.timeDiffCutoff))
    
    num_frames = len(avgTempBox)
    print ("len(avgTempBox) is " + str(num_frames))
    
    current_frame = 0
    end_axis = 0
    
    '''while current_frame < num_frames:
        
        if (current_frame < max_display_frames):
            start_frame = 0
            end_frame = current_frame
            end_axis = start_frame + max_display_frames
            frameSpeedStart = 0
            frameSpeedEnd = int(totalTime[current_frame] / 1000 * 30)
            startFrameSp = 0
            
        else:
            start_frame = current_frame - max_display_frames
            end_frame = current_frame
            end_axis = end_frame
            frameSpeedEnd = int(totalTime[current_frame] / 1000 * 30)
            frameSpeedStart = frameSpeedEnd - 3000
            startFrameSp = frameSpeedEnd - max_display_frames * 30
                
        plt.figure(figsize=(10, 6))
        
        plt.subplot(2, 1, 1)
        plt.ylim(0, 40)
        plt.xlim(startFrameSp, frameSpeedEnd + 30)
        plt.plot(speed[0:frameSpeedEnd], color='blue', label='Speed')
        plt.xlabel('Seconds')
        plt.ylabel('Speed')
        plt.title(f'Speed Graph (Frames {start_frame * 30}-{end_frame*30})')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(avgTempBox[0:end_frame], color='red', label='Average Temperature')
        plt.ylim(31.3, 32)
        plt.xlim(start_frame, end_axis)
        plt.xlabel('Seconds')
        plt.ylabel('Average Temperature')
        plt.title(f'Average Temperature Graph (Frames {start_frame}-{end_frame})')
        plt.legend()

        plt.tight_layout()
        
        name = mouse122.outputPath + 'mouse122_%05d.png' % current_frame
        #plt.savefig(name)
        
        #plt.show()

        current_frame += 1'''
    
    
    #Average Speed Overall, During, After
    #
    #
    
    counterLight = 0
    totalSpeed = 0
    totalSpeedDuring = 0
    numDuring = 0
    totalSpeedAfter = 0
    numAfter = 0
    
    afterSpeedFrames = []
    duringSpeedFrames = []
    num_frames = len(speed)
    print("len(speed) is " + str(num_frames))
    
    print("Light On Frames: " + str(mouse122.lightOnFrames))
    
    
    current_frame_temp = 0
    current_frame_speed = 0
    totalTime = 0
    cur_time = 0
    
    print("len(mouse122.avgTempBox) " + str(len(mouse122.avgTempBox)))
    print("len(mouse122.timeDifferences) " + str(len(mouse122.timeDifferences)))
    
    
    while current_frame_temp < len(mouse122.timeDifferences):
        
        totalTime += mouse122.timeDifferences[current_frame_temp] / 1000
        while (cur_time < totalTime and current_frame_speed < len(speed)):
            cur_time += 1/30
            
            if (counterLight < 8 and current_frame_temp > mouse122.lightOnFrames[counterLight] and current_frame_temp < mouse122.lightOnFrames[counterLight + 1]):
                totalSpeedDuring += speed[current_frame_speed]
                numDuring += 1
                duringSpeedFrames.append(speed[current_frame_speed])
                
            
            if (counterLight < 8 and current_frame_temp > mouse122.lightOnFrames[counterLight + 1] and current_frame_temp < mouse122.lightOnFrames[counterLight + 1] + 21):
                totalSpeedAfter += speed[current_frame_speed]
                numAfter += 1
                afterSpeedFrames.append(speed[current_frame_speed])
                
            if (counterLight < 8 and current_frame_temp > mouse122.lightOnFrames[counterLight + 1] + 21):
                counterLight += 2
            
            totalSpeed += speed[current_frame_speed]
            current_frame_speed += 1
            
        
        current_frame_temp += 1
        
        
    print("current_frame_speed: " + str(current_frame_speed))
    totalSpeed /= current_frame_speed
    totalSpeedDuring /= numDuring
    totalSpeedAfter /= numAfter
    
    print("Number of Frames during is: " + str(numDuring))
    print("Number of Frames after is: " + str(numAfter))
    
    
    sem_speed = np.std(speed[:num_frames]) / np.sqrt(num_frames)
    sem_speed_during = np.std(duringSpeedFrames) / np.sqrt(numDuring)
    sem_speed_after = np.std(afterSpeedFrames) / np.sqrt(numAfter)
        
    # Names for the bars
    variables = ['Average Speed', 'Average Speed During', 'Average Speed After']
    
    # Values for the bars
    values = [totalSpeed, totalSpeedDuring, totalSpeedAfter]
    
    # Create the bar chart
    plt.bar(variables, values, yerr=[sem_speed, sem_speed_during, sem_speed_after], color=['blue', 'green', 'red'], capsize=5)
    #plt.bar(variables, values, color=['blue', 'green', 'red'])
    
    # Add labels and title
    plt.ylabel('pixels/second')
    plt.title('Average Speed: Overall, During, and After Activations')
    
    # Display the plot
    plt.show()
    
    print("len(avgTempBox) is " + str(len(avgTempBox)))    

def graphSpeedvsPicture(speed):
    # Load video
    video_path = "/Users/david/Documents/Programming_Projects/CaltechProject/Files/mouseTemp_Recordings/mouse122/mouse122_1_hzcontBody.avi"
    cap = cv2.VideoCapture(video_path)
    
    # Check if video file is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    
    # Create a figure for the subplots
    plt.figure(figsize=(10, 6))
    
    # Loop through video frames
    frame_counter = 0
    frames_to_skip = 15
    
    while True:
        ret, frame = cap.read()
        
    
        if not ret:
            break
    
        # Display the frame and speed subplot side by side
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title(f"Frame {frame_counter}")
        plt.axis('off')
    
        lowerLimit = 0
        upperLimit = 0
        if frame_counter <= 3000:
            lowerLimit = 0
            upperLimit = frame_counter
        else:
            lowerLimit = frame_counter - 3000
            upperLimit = frame_counter
    
        plt.subplot(1, 2, 2)
        plt.plot(speed[lowerLimit:upperLimit])
        plt.xlim(0, 3000)
        plt.ylim(0, 40)
        plt.title("Speed")
        plt.ylabel("Pixels per Second")
    
        # Adjust layout and spacing between subplots
        plt.tight_layout()
    
        # Show the subplots
        plt.show()
        
        
        name = mouse122.outputPath + 'mouse122_%05d.png' % frame_counter
        #plt.savefig(name, dpi = 50)
        #plt.close()
        
        for _ in range(frames_to_skip - 1):
            cap.read()
            frame_counter += 1
        
        frame_counter += 1
    
        # Break the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the video capture object
    #cap.release()

def avgPupilAreaLineGraph(mouse_objects):
    x_values = range(0, 605 + 1801)
    x_values = [x / 30 for x in x_values]
    
    avgPupilArea = mouse_objects[0].avgPupilAreaAllTrials 
    stdPupilArea = mouse_objects[0].squarePupilAreasSum
    for mouse in mouse_objects[1:]:
        avgPupilArea += mouse.avgPupilAreaAllTrials
        stdPupilArea += mouse.squarePupilAreasSum
    
    '''for mouse in mouse_objects:
        print("Length: " + str(len(mouse.avgPupilAreaAllTrials)))
    
    avgPupilArea = np.empty((0, len(mouse_objects[0].avgPupilAreaAllTrials)))
    stdPupilArea = np.empty((0, len(mouse_objects[0].avgPupilAreaAllTrials)))
    
    for mouse in mouse_objects:
        avgPupilArea = np.vstack([avgPupilArea, mouse.avgPupilAreaAllTrials])
        stdPupilArea = np.vstack([avgPupilArea, mouse.squarePupilAreasSum])'''

    avgPupilArea /= len(mouse_objects)
    stdPupilArea = stdPupilArea/(len(mouse_objects) * 8 - 1)
    stdPupilArea -= np.square(avgPupilArea) * (len(mouse_objects) * 8) / (len(mouse_objects) * 8 - 1)
    stdPupilArea = np.sqrt(stdPupilArea) / np.sqrt((len(mouse_objects) * 8))
    
    smooth_pupil_areas, _ = smooth(avgPupilArea)
    
    laser_activation_patch = plt.axvspan(30, (605)/30 + 30, color=(1.0, 0.8, 0.8), alpha=0.5, label = "Laser Activation Period")
    plt.axvline(x=30, color='red', linestyle='dashed', alpha=0.5)
    plt.axvline(x=(605)/30 + 30, color='red', linestyle='dashed', alpha=0.5)
    
    #plt.axvline(x=self.cameraFrameNumbers[0]/30, color='red', linestyle='dashed', alpha=0.5)
    #plt.axvline(x=self.cameraFrameNumbers[1]/30, color='red', linestyle='dashed', alpha=0.5)
    
    ###Edits
    #
    #
    smooth_pupil_areas = smooth_pupil_areas[:2390]
    x_values = x_values[:2390]
    stdPupilArea = stdPupilArea[:2390]
    #
    #
    #
    
    plt.plot(x_values, smooth_pupil_areas, label='Pupil Area')
    plt.fill_between(x_values, smooth_pupil_areas - stdPupilArea, smooth_pupil_areas + stdPupilArea, alpha=0.5, label='SEM', color='gray')
    #plt.errorbar(x_values, smooth_pupil_areas, yerr=stdPupilArea, label='Pupil Area with SEM', color='b', ecolor='gray', capsize=5)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Pupil Area (Pixels^2)')
    plt.title('Average Pupil Area Over Time for Laser Activation (Mouse 38 & Mouse 48)')
    plt.legend()
    plt.show()
    
    
    

#General Functions
def smooth(area, win=30):
    """replace outliers in pupil area with smoothed pupil area"""
    """ also replace nan's with smoothed pupil area """
    """ smoothed pupil area (marea) is a median filter with window 30 """
    """ win = 30  usually recordings are @ >=30Hz """
    nt = area.size
    marea = np.zeros((win, nt))
    winhalf = int(win / 2)
    for k in np.arange(-winhalf, winhalf, 1, int):
        if k < 0:
            marea[k + winhalf, :k] = area[-k:]
        elif k > 0:
            marea[k + winhalf, k:] = area[:-k]
        else:
            marea[k + winhalf, :] = area
    marea = np.nanmedian(marea, axis=0)
    ix = np.logical_or(np.isnan(area), np.isnan(marea)).nonzero()[0]
    ix2 = (np.logical_and(~np.isnan(area), ~np.isnan(marea))).nonzero()[0]
    if ix2.size > 0:
        area[ix] = np.interp(ix, ix2, marea[ix2])
    else:
        area[ix] = 0
    marea[ix] = area[ix]

    # when do large deviations happen
    adiff = np.abs(area - marea)
    thres = area.std() / 2
    ireplace = adiff > thres
    area[ireplace] = marea[ireplace]

    return area, ireplace


#Main Code


    



#Declare Mouse Objects
#
#

### EXAMPLE MOUSE_OBJECT

#Mouse 121
filename = "/Users/david/Documents/Programming_Projects/CaltechProject/Files/mouseTemp_Recordings/mouse121/mouse121_fbf_1hz"
pupilPath = "/Users/david/Documents/Programming_Projects/CaltechProject/Files/mouseTemp_Recordings/mouse121/mouse121_1hzCont_proc.npy"
bodyPath = "/Users/david/Documents/Programming_Projects/CaltechProject/Files/mouseTemp_Recordings/mouse121/mouse121_1_hzcontBody.avi"
outputPath = "/Users/david/Documents/Programming_Projects/CaltechProject/VideoImages/mouse121/"
videoPath = "/Users/david/Documents/Programming_Projects/CaltechProject/Files/mouseTemp_Recordings/mouse121/mouse121_1hzCont.avi"
imageName = "mouse121frame"

frameNumbers = [1834, 2437, 6063, 6668, 10297, 10902, 14544, 15149, 18799, 19404, 23024, 23629, 27308, 27914, 31511, 32116]

#Eye
center_row = 105
center_col = 253


mouse121 = Mouse(filename, pupilPath, videoPath, outputPath, imageName, frameNumbers, center_row, center_col, "mouse121", bodyPath, 30)
print("Mouse 121 Created")
mouse121.openFile()
mouse121.getData()
#mouse121.plotAverageTempBox()
#mouse121.plotAverageTempOverall()
#mouse121.plotMaxTemp()
mouse121.plotEachTrial()
mouse121.plotAverageofTrials()
#mouse121.plotLineChartavgTempBeforeDuringAfter()
#mouse121.plotPupilSize()
#mouse121.saveDualGraph(1)
#



