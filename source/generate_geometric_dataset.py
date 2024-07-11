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

import numpy as np
import astra
from PIL import Image
import time
import numpy as np
import os

def create_geometries(height, width, numPixels, pixDistance, proj, distance):
    # Create volume and projection geometry
    vol_geom = astra.create_vol_geom(height, width)
    proj_geom = astra.create_proj_geom('fanflat', pixDistance, numPixels, np.linspace(0,2*np.pi,proj,False), distance, distance)
    return vol_geom, proj_geom

def create_sinogram(image, vol_geom, proj_geom):
    # Create a sinogram from the image
    proj_id = astra.create_projector('strip_fanflat',proj_geom,vol_geom)
    sinogram_id, sinogram = astra.create_sino(image, proj_id)
    # Clean up memory
    astra.data2d.delete(sinogram_id)
    return sinogram

t = time.time()

# Parameters
n = 1000
numPixels = 128 #256
pixDistance = 2.5 #5.0
proj = 128      #360
distance = 500
height, width = 128, 128

# Create geometries
vol_geom, proj_geom = create_geometries(height, width, numPixels, pixDistance, proj, distance)

# Project images
for index in range(n):
    # Open and save groundtruth in numpy format
    image = Image.open('dataset/groundtruth/gt_' + str(index) + '.png')
    image = np.float32(np.array(image))
    image = image / np.max(image)
    if not os.path.exists('datasetNotRescaled/groundtruth/'):
        os.makedirs('datasetNotRescaled/groundtruth/')
    np.save('datasetNotRescaled/groundtruth/gt_' + str(index) + '.npy', image)
    
    # Create and save the full sinogram
    sinogram = create_sinogram(image, vol_geom, proj_geom)
    if not os.path.exists('datasetNotRescaled/sinogram360/'):
        os.makedirs('datasetNotRescaled/sinogram360/')
    np.save('datasetNotRescaled/sinogram360/sino_' + str(index) + '.npy', sinogram)
    saved = sinogram
        
    # Eliminate projections (45 remaining)
    sinogram = saved
    for i in [j for j in [*range(0,proj)] if j not in [*range(0,proj,8)]]:
        sinogram[i,:] = 0
    if not os.path.exists('datasetNotRescaled/sinogram45/sino45_'):
        os.makedirs('datasetNotRescaled/sinogram45/sino45_')
    np.save('datasetNotRescaled/sinogram45/sino45_' + str(index) + '.npy', sinogram)

elapsed = time.time() - t
print('Elapsed seconds: ' + str(elapsed))



