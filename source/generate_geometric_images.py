# MIT License
#
# Copyright (c) 2024 C. Pezzoli, M. B. M. Paracchini, M. Marcon, S. Tubaro
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# In addition, any publications or presentations that use this software must
# include a citation to the following article:
#
# Title: Sinogram-Based Tomography Densification and Denoising based on a Physics-Informed Deep Learning Approach
# Authors: C. Pezzoli, M. B. M. Paracchini, M. Marcon, S. Tubaro
# Journal: IEEE TRANSACTIONS ON MEDICAL IMAGING
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import cv2
import numpy as np
import random
import time
import os 

t = time.time()

# Total number of images
n = 1000

# Image size
param = 4
img_size = (int(512/param), int(512/param))

# Define shapes
shapes = ['rectangle', 'triangle', 'circle', 'ellipse']

# Define image boarder margin
margin = int(30/param)
center_pos = int(160/param)

# Loop for images generation
for index in range(n):

    # Generate grayscale image
    img = np.zeros(img_size, dtype=np.uint8)

    # Define colors
    candidate_colors = [*range(50,256)]

    # Loop over the number of shapes
    shapes_number = random.randint(1, 3)

    for i in range(shapes_number):
        # Randomly select a shape
        shape = random.choice(shapes)
        
        # Get random color and position for the shape
        color = random.choice(candidate_colors)
        if i == shapes_number-1:
            color = 255
        remove_col = [*range(color-30, color+30)]
        candidate_colors = [col for col in candidate_colors if col not in remove_col]
        x, y = random.randint(0+center_pos, img_size[0]-center_pos), random.randint(0+center_pos, img_size[1]-center_pos)
        thickness = -1
        
        # Draw the shape
        if shape == 'rectangle':
            width, height = random.randint(int(50/param), int(200/param)), random.randint(int(50/param), int(200/param))
            x1, y1 = x - width//2, y - height//2
            x2, y2 = x + width//2, y + height//2
            x1, y1 = max(0 + margin, x1), max(0 + margin, y1)
            x2, y2 = min(img_size[0] - margin, x2), min(img_size[1] - margin, y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        elif shape == 'triangle':
            rd1 = random.randint(int(60/param),int(150/param))
            rd2 = random.randint(int(60/param),int(150/param))
            rd3 = random.randint(int(60/param),int(150/param))
            x1, x2, x3 = x-rd1, x+rd2, x-rd3
            y1, y2, y3 = y-rd1, y-rd2, y+rd3
            x1, x2, x3 = max(0+margin, x1), max(0+margin, x2), max(0+margin, x3)
            y1, y2, y3 = max(0+margin, y1), max(0+margin, y2), max(0+margin, y3)
            x1, x2, x3 = min(img_size[0] - margin, x1), min(img_size[0] - margin, x2), min(img_size[0] - margin, x3)
            y1, y2, y3 = min(img_size[1] - margin, y1), min(img_size[1] - margin, y2), min(img_size[1] - margin, y3)
            vertices = np.array([[x1, y1], [x2, y2], [x3, y3]], np.int32)
            cv2.fillPoly(img, [vertices], color)
        elif shape == 'circle':
            radius = random.randint(int(30/param), int(120/param))
            x1, y1 = x - radius, y - radius
            x2, y2 = x + radius, y + radius
            x1, y1 = max(0 + margin, x1), max(0 + margin, y1)
            x2, y2 = min(img_size[0] - margin, x2), min(img_size[1] - margin, y2)
            radius = min([x-x1, x2-x, y-y1, y2-y])
            center = (x, y)
            cv2.circle(img, center, radius, color, thickness)
        elif shape == 'ellipse':
            axes = (random.randint(int(100/param), int(160/param)), random.randint(int(30/param), int(80/param)))
            x1, x2, x3, x4 = x - axes[0], x + axes[0], x - axes[1], x + axes[1]
            y1, y2, y3, y4 = y - axes[0], y + axes[0], y - axes[1], y + axes[1]
            x1, x2, x3, x4 = max(0 + margin, x1), min(img_size[0] - margin, x2), max(0 + margin, x3), min(img_size[0] - margin, x4)
            y1, y2, y3, y4 = max(0 + margin, y1), min(img_size[0] - margin, y2), max(0 + margin, y3), min(img_size[0] - margin, y4)
            ax1 = max([x-x1, x2-x, x-x3, x4-x])
            ax2 = min([y-y1, y2-y, y-y3, y4-y])
            axes = (ax1, ax2)
            angle = random.randint(0, 90)
            center = (x, y)
            cv2.ellipse(img, center, axes, angle, 0, 360, color, thickness)

    # Save the image
    if not os.path.exists("dataset/groundtruth/"):
        os.makedirs("dataset/groundtruth/")
    cv2.imwrite("dataset/groundtruth/gt_" + str(index) + ".png", img)
    #img2 = cv2.imread(r'test.png', cv2.IMREAD_UNCHANGED)
    #cv2.imshow('Shapes', img2)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
elapsed = time.time() - t
print('Elapsed seconds: ' + str(elapsed))