#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 17:42:35 2023

@author: david
"""

'''import os
import imageio.v2 as imageio

os.environ["IMAGEIO_FFMPEG_EXE"] = "/Users/david/miniconda/envs/DEEPLABCUT_M1/bin/ffmpeg"

def create_video_from_images(image_folder_path, output_video_path, fps=60):
    images = []
    for filename in sorted(os.listdir(image_folder_path)):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            images.append(imageio.imread(os.path.join(image_folder_path, filename)))

    imageio.mimsave(output_video_path, images, fps=fps)
    
    # Set macro_block_size to 1 to prevent resizing
    #imageio.mimsave(output_video_path, images, fps=fps, macro_block_size=1)

# Example usage
print("Creation Beginning")
image_folder_path = '/Users/david/Documents/Programming_Projects/CaltechProject/VideoImages/mouse122'
output_video_path = '/Users/david/Documents/Programming_Projects/CaltechProject/speedvspictureFinal.mp4'
create_video_from_images(image_folder_path, output_video_path, fps = 60)

print("videoCreated") '''


import os
import imageio.v2 as imageio

os.environ["IMAGEIO_FFMPEG_EXE"] = "/Users/david/miniconda/envs/DEEPLABCUT_M1/bin/ffmpeg"

def create_video_from_images(image_folder_path, output_video_path, fps=5):
    images = []
    for filename in sorted(os.listdir(image_folder_path)):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            im = imageio.imread(os.path.join(image_folder_path, filename))
            images.append(im)

    imageio.mimsave(output_video_path, images, fps=fps)
    
    # Set macro_block_size to 1 to prevent resizing
    #imageio.mimsave(output_video_path, images, fps=fps, macro_block_size=1)

# Example usage
print("Creation Beginning")
image_folder_path = '/Users/david/Documents/Programming_Projects/CaltechProject/VideoImages/mouse38_fear/'
output_video_path = '/Users/david/Documents/Programming_Projects/CaltechProject/Data Videos/Sf1_Example.mp4'
create_video_from_images(image_folder_path, output_video_path, fps = 5)

print("videoCreated")
