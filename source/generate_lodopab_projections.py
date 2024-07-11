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

import h5py
import matplotlib.pyplot as plt
import numpy as np
import astra
from skimage.transform import resize
from odl.discr import uniform_partition
from odl import uniform_discr
from math import ceil
from itertools import islice
import odl
from tqdm import tqdm
import multiprocessing
import os 

# Parameters
MU_WATER = 20
MU_AIR = 0.02
MU_MAX = 3071 * (MU_WATER - MU_AIR) / 1000 + MU_WATER
PHOTONS_PER_PIXEL = 4096
MIN_PT = [-0.13, -0.13]
MAX_PT = [0.13, 0.13]
NUM_DET_PIXELS = 513
IM_SHAPE = (1000, 1000)
RECO_IM_SHAPE = (362, 362)
PROJ = 1000
rs = np.random.RandomState(3)

# Functions
def create_geometries():
    # Create parameters compatible with odl.tomo
    angle_partition = uniform_partition(0, np.pi, PROJ)
    angles = angle_partition.coord_vectors[0] - np.pi / 2
    space = uniform_discr(min_pt=MIN_PT, max_pt=MAX_PT, shape=IM_SHAPE, dtype=np.float64)
    rho = np.max(np.linalg.norm(space.domain.corners()[:, :2], axis=1))
    det_partition = uniform_partition(-rho, rho, (NUM_DET_PIXELS,))
    det_width = det_partition.cell_sides[0]
    vol_shp = space.partition.shape
    vol_min = space.partition.min_pt
    vol_max = space.partition.max_pt
    # Create geometries
    vol_geom = astra.create_vol_geom(vol_shp[0], vol_shp[1], vol_min[1], vol_max[1], -vol_max[0], -vol_min[0])
    proj_geom = astra.create_proj_geom('parallel', det_width, NUM_DET_PIXELS, angles)
    return vol_geom, proj_geom

def create_sinogram(image, vol_geom, proj_geom):
    proj_id = astra.create_projector('linear',proj_geom,vol_geom)
    sinogram_id, sinogram = astra.create_sino(image, proj_id, gpuIndex=0)
    astra.data2d.delete(sinogram_id)
    return sinogram

def forward_fun(im):
    # Create and save the full sinogram
    sinogram = create_sinogram(resize(im*MU_MAX, IM_SHAPE, order=1), vol_geom, proj_geom)
    sinogram *= (-1)
    np.exp(sinogram, out=sinogram)
    sinogram *= PHOTONS_PER_PIXEL
    sinogram = sinogram / PHOTONS_PER_PIXEL
    np.maximum(0.1 / PHOTONS_PER_PIXEL, sinogram, out=sinogram)
    np.log(sinogram, out=sinogram)
    sinogram /= (-MU_MAX)
    return sinogram
    
# Create geometries
vol_geom, proj_geom = create_geometries()

# Other declarations
reco_space = odl.uniform_discr(min_pt=MIN_PT, max_pt=MAX_PT,shape=RECO_IM_SHAPE, dtype=np.float32)
space = odl.uniform_discr(min_pt=MIN_PT, max_pt=MAX_PT, shape=IM_SHAPE, dtype=np.float64)
geometry = odl.tomo.parallel_beam_geometry(space, num_angles=PROJ, det_shape=(NUM_DET_PIXELS,))
ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cpu')
# for gt sino and sparse sino
NUM_SAMPLES_PER_FILE = 128
PATH = '/lodopab'
LEN = {
    'train': 35820,
    'validation': 3522,
    'test': 3553}

def ground_truth_gen(part, start):
    num_files = ceil(LEN[part] / NUM_SAMPLES_PER_FILE)
    for i in range(start, num_files):
        with h5py.File('lodopab/ground_truth_{}/ground_truth_{}_{:03d}.hdf5'.format(part, part, i), 'r') as file:
            ground_truth_data = file['data'][:]
        for gt_arr in ground_truth_data:
            yield reco_space.element(gt_arr)
            
for part in ['train', 'validation', 'test']:
    if part == 'train':
        start = 70
    else:
        start = 5
    gen = ground_truth_gen(part, start)
    n_files = ceil(LEN[part] / NUM_SAMPLES_PER_FILE)
    for filenumber in tqdm(range(start, n_files), desc=part):
        gt_filename = 'lodopab/gt_sino_{}/gt_sino_{}_{:03d}.hdf5'.format(part, part, filenumber)
        if not os.path.exists('lodopab/gt_sino_{}'.format(part)):
            os.makedirs('lodopab/gt_sino_{}'.format(part))
        with h5py.File(gt_filename, 'w') as gt_file:
            gt_sino_dataset = gt_file.create_dataset(
                'data', shape=(NUM_SAMPLES_PER_FILE,) + ray_trafo.range.shape,
                maxshape=(NUM_SAMPLES_PER_FILE,) + ray_trafo.range.shape,
                dtype=np.float32, chunks=True)

            im_buf = [im for im in islice(gen, NUM_SAMPLES_PER_FILE)]
            
            with multiprocessing.Pool(20) as pool:
                gt_buf= pool.map(forward_fun, im_buf)
                
            for i, (im, gt) in enumerate(zip(im_buf, gt_buf)):
                gt_sino_dataset[i] = gt
                
            # resize last file
            if filenumber == n_files - 1:
                gt_sino_dataset.resize(LEN[part] - (n_files - 1) * NUM_SAMPLES_PER_FILE,axis=0)
# For sparse sino with noise
NUM_SAMPLES_PER_FILE = 128
PATH = '/localdata/lodopab'
LEN = {
    'train': 35820,
    'validation': 3522,
    'test': 3553}

def observation_gen(part):
    num_files = ceil(LEN[part] / NUM_SAMPLES_PER_FILE)
    for i in range(num_files):
        with h5py.File('lodopab/observation_{}/observation_{}_{:03d}.hdf5'.format(part, part, i), 'r') as file:
            observation_data = file['data'][:]
        for obs_arr in observation_data:
            yield obs_arr
            
for part in ['train', 'validation', 'test']:
    if part == 'train':
        continue
    gen = observation_gen(part)
    n_files = ceil(LEN[part] / NUM_SAMPLES_PER_FILE)
    for filenumber in tqdm(range(n_files), desc=part):
        if (part =='train' and filenumber == 70) or (part !='train' and filenumber == 5):
            break
        noisy_sparse_filename = 'lodopab/noisy_sparse_sino_{}/noisy_sparse_sino_{}_{:03d}.hdf5'.format(part, part, filenumber)
        if not os.path.exists('lodopab/noisy_sparse_sino_{}'.format(part)):
            os.makedirs('lodopab/noisy_sparse_sino_{}'.format(part))
        with h5py.File(noisy_sparse_filename, 'w') as ns_file:
            noisy_sparse_sino_dataset = ns_file.create_dataset(
                'data', shape=(NUM_SAMPLES_PER_FILE,) + ray_trafo.range.shape,
                maxshape=(NUM_SAMPLES_PER_FILE,) + ray_trafo.range.shape,
                dtype=np.float32, chunks=True)
            im_buf = [im for im in islice(gen, NUM_SAMPLES_PER_FILE)]                
            for i, im in enumerate(im_buf):
                noisy_sparse = np.copy(im)
                for k in [j for j in [*range(0,PROJ)] if j not in [*range(0,PROJ,8)]]:
                    noisy_sparse[k,:] = 0
                noisy_sparse_sino_dataset[i] = noisy_sparse
                
            # resize last file
            if filenumber == n_files - 1:
                noisy_sparse_sino_dataset.resize(LEN[part] - (n_files - 1) * NUM_SAMPLES_PER_FILE,axis=0)
