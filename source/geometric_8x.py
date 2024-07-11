import random, datetime, time, os
import numpy as np
import matplotlib.pyplot as plt
import astra
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from math import sqrt
import nvidia_smi

import tensorflow as tf
from IPython import display

from keras import backend as K
from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation, Dropout, concatenate, LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import load_img, img_to_array

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(physical_devices)


nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(0, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))
while True:
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    if 100*info.free/info.total > 50:
        break
    else:
        time.sleep(10)
print('GPU is now free.')
nvidia_smi.nvmlShutdown()

# Some parameters

name = "dual-gan-full"
tune = 5
lrt = 2e-4 #* sqrt(tune)
ls = "generator_mse"
bsz = 1 * tune
STEPS = 120000

parameters = {
    "pixelLengthX" : 1.0,
    "pixelLengthY" : 1.0,
    "colCount": 128,
    "rowCount" : 128,
    "windowMinX" : -64,
    "windowMaxX": 64,
    "windowMinY" : -64,
    "windowMaxY": 64,
    "detSpacing": 2.5,
    "detCount" : 128,
    "anglesVec" : np.linspace(0, np.pi, 64, False),
    "sourceDetDist": 500,
    "proj": 64
}

maximumImage = np.ones((parameters["rowCount"], parameters["colCount"]))
vol_geom = astra.create_vol_geom(parameters["rowCount"], parameters["colCount"], 
                                     parameters["windowMinX"], parameters["windowMaxX"], 
                                     parameters["windowMinY"], parameters["windowMaxY"])
proj_geom = astra.create_proj_geom('fanflat', parameters["detSpacing"], parameters["detCount"], parameters["anglesVec"], parameters["sourceDetDist"],parameters["sourceDetDist"])
proj_id = astra.create_projector('strip_fanflat', proj_geom, vol_geom)
doNotCare, maxSinogram = astra.create_sino(maximumImage, proj_id)
maxValue = np.max(maxSinogram)

# Dataset loading

def load(image_number):
    # Sparse sinogram
    input_sino = np.load("datasetNotRescaled/sinogram45/sino45_" + str(image_number) + ".npy")
    input_sino = input_sino[0:64,:]
    input_sino = (input_sino / maxValue)# * 2) - 1
    input_sino = tf.convert_to_tensor(input_sino)
    input_sino = tf.expand_dims(input_sino, -1)
    # Full sinogram
    real_sino = np.load("datasetNotRescaled/sinogram360/sino_" + str(image_number) + ".npy")
    real_sino = real_sino[0:64,:]
    real_sino = (real_sino / maxValue)# * 2) - 1
    real_sino = tf.convert_to_tensor(real_sino)
    real_sino = tf.expand_dims(real_sino, -1)
    # GT image
    real_image = np.load("datasetNotRescaled/groundtruth/gt_" + str(image_number) + ".npy")
    real_image = tf.convert_to_tensor(real_image)
    real_image = tf.expand_dims(real_image, -1)
    return input_sino, real_sino, real_image


BUFFER_SIZE = 1000
full_dataset = tf.data.Dataset.from_tensors(load(0))
for i in range(1,BUFFER_SIZE):
    current_dataset = tf.data.Dataset.from_tensors(load(i))
    full_dataset = full_dataset.concatenate(current_dataset)


BATCH_SIZE = bsz
full_dataset = full_dataset.shuffle(BUFFER_SIZE, seed=42)
print(full_dataset.reduce(np.int64(0), lambda x, _: x + 1).numpy())
train_dataset = full_dataset.take(900)
train_dataset = train_dataset.batch(BATCH_SIZE)
print(train_dataset.reduce(np.int64(0), lambda x, _: x + 1).numpy())
test_dataset = full_dataset.skip(900)
test_dataset = test_dataset.batch(BATCH_SIZE)
print(test_dataset.reduce(np.int64(0), lambda x, _: x + 1).numpy())


# Projection utilities

def make_projection_matrix(parameters):
    vol_geom = astra.create_vol_geom(parameters["rowCount"], parameters["colCount"], 
                                     parameters["windowMinX"], parameters["windowMaxX"], 
                                     parameters["windowMinY"], parameters["windowMaxY"])
    proj_geom = astra.create_proj_geom('fanflat', parameters["detSpacing"], parameters["detCount"], parameters["anglesVec"], parameters["sourceDetDist"],parameters["sourceDetDist"])
    proj_id = astra.create_projector('strip_fanflat', proj_geom, vol_geom)
    matrix_id = astra.projector.matrix(proj_id)
    projMatrix = astra.matrix.get(matrix_id)
    projMatrixTf=projMatrix.todense()
    projMatrixTf = np.float32(projMatrixTf)
    projMatrixTensorflow = tf.convert_to_tensor(projMatrixTf, dtype=tf.float32)
    return projMatrixTensorflow

# Sinogram generator and discriminator definition

class ReplaceLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(ReplaceLayer, self).__init__()
        replaceMask = np.full((64, 128, 1), False)
        for i in [j for j in [*range(0,64)] if j not in [*range(0,64,8)]]:
            replaceMask[i,:,:]=True
        self.replaceMaskTensor = tf.Variable(initial_value = tf.convert_to_tensor(replaceMask, dtype=tf.bool), trainable=False, dtype=tf.bool, shape=[64, 128, 1])

    def call(self, output_tensor, input_tensor):
        x = tf.where(self.replaceMaskTensor, output_tensor, input_tensor)
        return x

def sinoGenerator():
    
    inputs = tf.keras.layers.Input(shape=[64, 128, 1])
    initializer = tf.random_normal_initializer(0., 0.02)
  
    # Contracting Path
    nFilt = 64
    kerSize = 5
    stride = 2
    
    c1 = Conv2D(filters = nFilt * 1, kernel_size=(kerSize, kerSize), strides=(stride, stride), padding = 'same',kernel_initializer=initializer)(inputs)
    c1 = Activation(LeakyReLU())(c1)
    
    c2 = Conv2D(filters = nFilt * 2, kernel_size=(kerSize, kerSize), strides=(stride, stride), padding = 'same',kernel_initializer=initializer)(c1)
    c2 = BatchNormalization()(c2)
    c2 = Activation(LeakyReLU())(c2)
    
    c3 = Conv2D(filters = nFilt * 4, kernel_size=(kerSize, kerSize), strides=(stride, stride), padding = 'same',kernel_initializer=initializer)(c2)
    c3 = BatchNormalization()(c3)
    c3 = Activation(LeakyReLU())(c3)
    
    c4 = Conv2D(filters = nFilt * 8, kernel_size=(kerSize, kerSize), strides=(stride, stride), padding = 'same',kernel_initializer=initializer)(c3)
    c4 = BatchNormalization()(c4)
    c4 = Activation(LeakyReLU())(c4)
    
    # Expansive Path
    u5 = Conv2DTranspose(nFilt * 4, (kerSize, kerSize), strides =(stride, stride), padding = 'same',kernel_initializer=initializer)(c4)
    u5 = concatenate([u5, c3])
    u5 = BatchNormalization()(u5)
    u5 = Activation('relu')(u5)
    
    u6 = Conv2DTranspose(nFilt * 2, (kerSize, kerSize), strides =(stride, stride), padding = 'same',kernel_initializer=initializer)(u5)
    u6 = concatenate([u6, c2])
    u6 = BatchNormalization()(u6)
    u6 = Activation('relu')(u6)
    
    u7 = Conv2DTranspose(nFilt * 1, (kerSize, kerSize), strides =(stride, stride), padding = 'same',kernel_initializer=initializer)(u6)
    u7 = concatenate([u7, c1])
    u7 = BatchNormalization()(u7)
    u7 = Activation('relu')(u7)
    
    u8 = Conv2DTranspose(1, (kerSize, kerSize), strides =(stride, stride), padding = 'same',kernel_initializer=initializer)(u7)
    u8 = tf.keras.activations.sigmoid(u8)
    outputs = ReplaceLayer()(u8, inputs)    

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def sinoDiscriminator():
    
    inputs = tf.keras.layers.Input(shape=[64, 128, 1]) #, batch_size=BATCH_SIZE
    initializer = tf.random_normal_initializer(0., 0.02)
  
    # Contracting Path
    nFilt = 16
    kerSize = 5
    stride = 2
    
    c1 = Conv2D(filters = nFilt * 1, kernel_size=(kerSize, kerSize), strides=(stride, stride), padding = 'same',kernel_initializer=initializer)(inputs)
    c1 = Activation(LeakyReLU())(c1)
    
    c2 = Conv2D(filters = nFilt * 2, kernel_size=(kerSize, kerSize), strides=(stride, stride), padding = 'same',kernel_initializer=initializer)(c1)
    c2 = BatchNormalization()(c2)
    c2 = Activation(LeakyReLU())(c2)
    
    c3 = Conv2D(filters = nFilt * 4, kernel_size=(kerSize, kerSize), strides=(stride, stride), padding = 'same',kernel_initializer=initializer)(c2)
    c3 = BatchNormalization()(c3)
    c3 = Activation(LeakyReLU())(c3)
    
    c4 = Conv2D(filters = nFilt * 8, kernel_size=(kerSize, kerSize), strides=(stride, stride), padding = 'same',kernel_initializer=initializer)(c3)
    c4 = BatchNormalization()(c4)
    c4 = Activation(LeakyReLU())(c4)
    
    # Expansive Path
    u5 = Conv2DTranspose(nFilt * 4, (kerSize, kerSize), strides =(stride, stride), padding = 'same',kernel_initializer=initializer)(c4)
    u5 = BatchNormalization()(u5)
    u5 = Activation('relu')(u5)
    
    u6 = Conv2DTranspose(nFilt * 2, (kerSize, kerSize), strides =(stride, stride), padding = 'same',kernel_initializer=initializer)(u5)
    u6 = BatchNormalization()(u6)
    u6 = Activation('relu')(u6)
    
    u7 = Conv2DTranspose(nFilt * 1, (kerSize, kerSize), strides =(stride, stride), padding = 'same',kernel_initializer=initializer)(u6)
    u7 = BatchNormalization()(u7)
    u7 = Activation('relu')(u7)
    
    u8 = Conv2DTranspose(1, (kerSize, kerSize), strides =(stride, stride), padding = 'same',kernel_initializer=initializer)(u7)
    outputs = tf.keras.activations.sigmoid(u8)

    return Model(inputs=[inputs], outputs=[outputs])

# Filtering utilities

def make_ram_lak(n):
    filt = np.zeros(n)
    filt[0] = 0.25
    odd_indices = np.arange(1, n, 2)
    cond = 2 * odd_indices > n
    odd_indices[cond] = n - odd_indices[cond]
    filt[1::2] = -1 / (np.pi * odd_indices) ** 2
    return tf.convert_to_tensor(filt, dtype=tf.float32)

def filter_sino(y, filt, padded=True):
    original_width = y.shape[-2]
    if padded:
        y = pad(y)
        if len(filt) == original_width:
            filt = pad_filter(filt)
    if filt.shape != y.shape[-2:-1]:
        raise ValueError("Filter wrong len!")
    y = tf.transpose(y, perm=[0, 3, 1, 2])
    y_f = tf.signal.rfft(y)
    h_f = tf.signal.rfft(filt)
    y_f *= h_f
    y_filtered = tf.signal.irfft(y_f, fft_length=[y.shape[-1]])
    y_filtered = tf.transpose(y_filtered, perm=[0, 2, 3, 1])
    if padded:
        y_filtered = unpad(y_filtered, original_width)
    return y_filtered

def pad(sino):
    (batch_size, num_angles, num_pixels, num_slices) = sino.shape
    num_pad_left, num_pad_right = num_pad(num_pixels)   
    paddings = tf.constant([[0, 0], [0, 0], [num_pad_left, num_pad_right], [0, 0]])
    sino = tf.pad(sino, paddings, mode='CONSTANT')
    return sino

def unpad(sino, width):
    pad_left, pad_right = num_pad(width)
    return sino[:,:, pad_left:pad_left+width,:]

def pad_filter(h):
    out= tf.reverse(h,[0])
    out = tf.roll(out, [1], [0])
    out=tf.concat([h,out],0)
    return out

def num_pad(width):
    num_padding = width
    num_pad_left = num_padding // 2
    return (num_pad_left, num_padding - num_pad_left)

# Projection utilities
def back_projection(sinogram, proj_matrix, img_pix, det_pix, n_proj):
    for i in range(bsz):
        flat_sinogram = tf.reshape(sinogram[i,:,:,:], (det_pix*n_proj, 1))
        transposed_proj_matrix = tf.transpose(proj_matrix, perm=[1, 0])
        backward = tf.matmul(transposed_proj_matrix, flat_sinogram)
        reshaped = tf.reshape(backward, [1, img_pix, img_pix, 1])
        if i == 0:
            final = reshaped
        else:
            final = tf.concat([final, reshaped], 0)
    return final

def back_projection_old(sinogram, proj_matrix, img_pix, det_pix, n_proj):
    flat_sinogram = tf.reshape(sinogram, (det_pix*n_proj, 1))
    transposed_proj_matrix = tf.transpose(proj_matrix, perm=[1, 0])
    backward = tf.matmul(transposed_proj_matrix, flat_sinogram)
    return tf.reshape(backward, [1, img_pix, img_pix, 1])

# Final image generator and discriminator definition
class RamLakFilterLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(RamLakFilterLayer, self).__init__()
        self.RamLak = make_ram_lak(parameters["detCount"])

    def call(self, sino):
        filtered_sino = filter_sino(sino, self.RamLak, padded=True)
        return filtered_sino
    
class BackProjectionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(BackProjectionLayer, self).__init__()
        sino_ones = tf.ones([parameters["proj"], parameters["detCount"]])
        self.sens_image = back_projection_old(sino_ones, projection_matrix, parameters["rowCount"], parameters["detCount"], parameters["proj"])

    def call(self, filtered_sino):
        rec = back_projection(filtered_sino, projection_matrix, parameters["rowCount"], parameters["detCount"], parameters["proj"]) / self.sens_image
        return rec

def recGenerator():
    
    inputs = tf.keras.layers.Input(shape=[64, 128, 1])
    initializer = tf.random_normal_initializer(0., 0.02)
    # Backprojection
    
    sinoRescaled = inputs * maxValue
    filtered_sino = RamLakFilterLayer()(sinoRescaled)
    noisyRec = BackProjectionLayer()(filtered_sino)
    noisyRec = Activation('relu')(noisyRec)
    # Contracting Path
    nFilt = 64
    kerSize = 3
    stride = 2
    
    c1 = Conv2D(filters = nFilt * 1, kernel_size=(kerSize, kerSize), strides=(stride, stride), padding = 'same',kernel_initializer=initializer)(noisyRec)
    c1 = Activation(LeakyReLU())(c1)
    
    c2 = Conv2D(filters = nFilt * 2, kernel_size=(kerSize, kerSize), strides=(stride, stride), padding = 'same',kernel_initializer=initializer)(c1)
    c2 = BatchNormalization()(c2)
    c2 = Activation(LeakyReLU())(c2)
    
    c3 = Conv2D(filters = nFilt * 4, kernel_size=(kerSize, kerSize), strides=(stride, stride), padding = 'same',kernel_initializer=initializer)(c2)
    c3 = BatchNormalization()(c3)
    c3 = Activation(LeakyReLU())(c3)
    
    c4 = Conv2D(filters = nFilt * 8, kernel_size=(kerSize, kerSize), strides=(stride, stride), padding = 'same',kernel_initializer=initializer)(c3)
    c4 = BatchNormalization()(c4)
    c4 = Activation(LeakyReLU())(c4)
    
    # Expansive Path
    u5 = Conv2DTranspose(nFilt * 4, (kerSize, kerSize), strides =(stride, stride), padding = 'same',kernel_initializer=initializer)(c4)
    u5 = concatenate([u5, c3])
    u5 = BatchNormalization()(u5)
    u5 = Activation('relu')(u5)
    
    u6 = Conv2DTranspose(nFilt * 2, (kerSize, kerSize), strides =(stride, stride), padding = 'same',kernel_initializer=initializer)(u5)
    u6 = concatenate([u6, c2])
    u6 = BatchNormalization()(u6)
    u6 = Activation('relu')(u6)
    
    u7 = Conv2DTranspose(nFilt * 1, (kerSize, kerSize), strides =(stride, stride), padding = 'same',kernel_initializer=initializer)(u6)
    u7 = concatenate([u7, c1])
    u7 = BatchNormalization()(u7)
    u7 = Activation('relu')(u7)
    
    u8 = Conv2DTranspose(1, (kerSize, kerSize), strides =(stride, stride), padding = 'same',kernel_initializer=initializer)(u7)
    outputs = Activation('relu')(u8)  
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def recDiscriminator():
    
    inputs = tf.keras.layers.Input(shape=[128, 128, 1]) 
    initializer = tf.random_normal_initializer(0., 0.02)
  
    # Contracting Path
    nFilt = 16
    kerSize = 5
    stride = 2
    
    c1 = Conv2D(filters = nFilt * 1, kernel_size=(kerSize, kerSize), strides=(stride, stride), padding = 'same',kernel_initializer=initializer)(inputs)
    c1 = Activation(LeakyReLU())(c1)
    
    c2 = Conv2D(filters = nFilt * 2, kernel_size=(kerSize, kerSize), strides=(stride, stride), padding = 'same',kernel_initializer=initializer)(c1)
    c2 = BatchNormalization()(c2)
    c2 = Activation(LeakyReLU())(c2)
    
    c3 = Conv2D(filters = nFilt * 4, kernel_size=(kerSize, kerSize), strides=(stride, stride), padding = 'same',kernel_initializer=initializer)(c2)
    c3 = BatchNormalization()(c3)
    c3 = Activation(LeakyReLU())(c3)
    
    c4 = Conv2D(filters = nFilt * 8, kernel_size=(kerSize, kerSize), strides=(stride, stride), padding = 'same',kernel_initializer=initializer)(c3)
    c4 = BatchNormalization()(c4)
    c4 = Activation(LeakyReLU())(c4)
    
    # Expansive Path
    u5 = Conv2DTranspose(nFilt * 4, (kerSize, kerSize), strides =(stride, stride), padding = 'same',kernel_initializer=initializer)(c4)
    u5 = concatenate([u5, c3])
    u5 = BatchNormalization()(u5)
    u5 = Activation('relu')(u5)
    
    u6 = Conv2DTranspose(nFilt * 2, (kerSize, kerSize), strides =(stride, stride), padding = 'same',kernel_initializer=initializer)(u5)
    u6 = concatenate([u6, c2])
    u6 = BatchNormalization()(u6)
    u6 = Activation('relu')(u6)
    
    u7 = Conv2DTranspose(nFilt * 1, (kerSize, kerSize), strides =(stride, stride), padding = 'same',kernel_initializer=initializer)(u6)
    u7 = concatenate([u7, c1])
    u7 = BatchNormalization()(u7)
    u7 = Activation('relu')(u7)
    
    u8 = Conv2DTranspose(1, (kerSize, kerSize), strides =(stride, stride), padding = 'same',kernel_initializer=initializer)(u7)
    outputs = tf.keras.activations.sigmoid(u8)

    return Model(inputs=[inputs], outputs=[outputs])


# SSIM definition

def extract_image_patches(x, ksizes, ssizes, padding='same', data_format='channels_last'):
    # Arguments: x (input image), ksizes (2-d tuple kernel size), ssizes (2-d tuple strides size), padding ('same' or 'valid'), data_format ('channels_last' or 'channels_first')
    # Returns The (k_w,k_h) patches extracted. TF ==> (batch_size,w,h,k_w,k_h,c) TH ==> (batch_size,w,h,c,k_w,k_h)

    kernel = [1, ksizes[0], ksizes[1], 1]
    strides = [1, ssizes[0], ssizes[1], 1]
    padding = padding.upper()
    if data_format == 'channels_first':
        x = K.permute_dimensions(x, (0, 2, 3, 1))
    bs_i, w_i, h_i, ch_i = K.int_shape(x)
    patches = tf.image.extract_patches(x, kernel, strides, [1, 1, 1, 1], padding)
    bs, w, h, ch = K.int_shape(patches)
    reshaped = tf.reshape(patches, [-1, w, h, tf.math.floordiv(ch, ch_i), ch_i])
    final_shape = [-1, w, h, ch_i, ksizes[0], ksizes[1]]
    patches = tf.reshape(tf.transpose(reshaped, [0, 1, 2, 4, 3]), final_shape)
    if data_format == 'channels_last':
        patches = K.permute_dimensions(patches, [0, 1, 2, 4, 5, 3])
    return patches

def CalculateDSSIM(y_true, y_pred, kernel_size=3, k1=0.01, k2=0.03, max_value=1.0):
    # DSSIM, clipped between 0.0 and 1.0. Arguments: k1: Parameter of the SSIM (default 0.01), k2: Parameter of the SSIM (default 0.03), 
    # kernel_size: Size of the sliding window (default 3), max_value: Dinamic Range of the image (default 1.0)

    # Bring tensors from range -1.0 - +1.0 in range 0.0 - +1.0
    #y_true = (y_true + 1.0) / 2
    #y_pred = (y_pred + 1.0) / 2

    # Define the kernel and constants
    kernel = [kernel_size, kernel_size]
    C1 = (k1 * max_value) ** 2
    C2 = (k2 * max_value) ** 2

    # Reshape the inputs to work with the sliding window
    y_true = K.reshape(y_true, [-1] + list(K.int_shape(y_pred)[1:]))
    y_pred = K.reshape(y_pred, [-1] + list(K.int_shape(y_pred)[1:]))

    # Extract image patches for y_true and y_pred
    patches_pred = extract_image_patches(y_pred, kernel, kernel, 'valid', K.image_data_format())
    patches_true = extract_image_patches(y_true, kernel, kernel, 'valid', K.image_data_format())

    # Reshape the extracted patches to calculate statistics on each patch
    bs, w, h, c1, c2, c3 = K.int_shape(patches_pred)
    patches_pred = K.reshape(patches_pred, [-1, w, h, c1 * c2 * c3])
    patches_true = K.reshape(patches_true, [-1, w, h, c1 * c2 * c3])

    # Get mean, varianche and std dev of the patches
    u_true = K.mean(patches_true, axis=-1)
    u_pred = K.mean(patches_pred, axis=-1)
    var_true = K.var(patches_true, axis=-1)
    var_pred = K.var(patches_pred, axis=-1)
    covar_true_pred = K.mean(patches_true * patches_pred, axis=-1) - u_true * u_pred
    # Compute SSIM
    ssim = (2 * u_true * u_pred + C1) * (2 * covar_true_pred + C2)
    denom = ((K.square(u_true) + K.square(u_pred) + C1) * (var_pred + var_true + C2))
    ssim /= denom
    return K.mean(1.0 - ssim)  

# Sinogram networks losses definition

S_LAMBDA_MSE = 100  
S_LAMBDA_SSIM = 0.5
S_LAMBDA_DISC = 10

comparing_matrix = np.zeros((BATCH_SIZE,64,128,1))
for i in [j for j in [*range(0,64)] if j in [*range(0,64,8)]]:
            comparing_matrix[:,i,:,:]=1
comparing_tensor = tf.convert_to_tensor(comparing_matrix, dtype=tf.float32)

def sino_generator_loss(gen_output, target):
    squaredTensor = tf.square(target - gen_output)
    l2_loss = tf.reduce_mean(squaredTensor)
    ssim_loss = CalculateDSSIM(target, gen_output, kernel_size=7, k1=0.01, k2=0.03, max_value=1.0)
    total_gen_loss = (S_LAMBDA_MSE * l2_loss) + (S_LAMBDA_SSIM * ssim_loss)
    return total_gen_loss, l2_loss, ssim_loss

def sino_generator_loss2(disc_generated_output, gen_output, target):
    squaredTensor = tf.square(tf.ones_like(disc_generated_output) - disc_generated_output)
    cheated_disc_loss = tf.reduce_mean(squaredTensor)
    squaredTensor = tf.square(target - gen_output)
    l2_loss = tf.reduce_mean(squaredTensor)
    ssim_loss = CalculateDSSIM(target, gen_output, kernel_size=7, k1=0.01, k2=0.03, max_value=1.0)
    total_gen_loss = (S_LAMBDA_DISC * cheated_disc_loss) + (S_LAMBDA_MSE * l2_loss) + (S_LAMBDA_SSIM * ssim_loss)
    return total_gen_loss, cheated_disc_loss, l2_loss, ssim_loss

def sino_discriminator_loss(disc_real_output, disc_generated_output):
    squaredTensor = tf.square(tf.ones_like(disc_real_output) - disc_real_output)
    real_recognition_loss = tf.reduce_mean(squaredTensor)
    real_accuracy = 100 - (tf.reduce_mean(tf.abs(tf.ones_like(disc_real_output) - disc_real_output)) * 100)
    
    squaredTensor = tf.square(tf.zeros_like(disc_generated_output) - disc_generated_output)
    generated_recognition_loss = tf.reduce_mean(squaredTensor)
    generated_accuracy = 100 - (tf.reduce_mean(tf.abs(tf.zeros_like(disc_generated_output) - disc_generated_output)) * 100)
    
    total_disc_loss = real_recognition_loss + generated_recognition_loss

    return total_disc_loss, real_recognition_loss, generated_recognition_loss, real_accuracy, generated_accuracy

# Reconstruction networks losses definition

R_LAMBDA_MSE = 100  
R_LAMBDA_SSIM = 1
R_LAMBDA_DISC = 10

def rec_generator_loss(gen_output, target):
    squaredTensor = tf.square(target - gen_output)
    l2_loss = tf.reduce_mean(squaredTensor)
    ssim_loss = CalculateDSSIM(target, gen_output, kernel_size=7, k1=0.01, k2=0.03, max_value=1.0)
    total_gen_loss = (R_LAMBDA_MSE * l2_loss) + (R_LAMBDA_SSIM * ssim_loss)
    return total_gen_loss, l2_loss, ssim_loss

def rec_generator_loss2(disc_generated_output, gen_output, target):
    squaredTensor = tf.square(tf.ones_like(disc_generated_output) - disc_generated_output)
    cheated_disc_loss = tf.reduce_mean(squaredTensor)
    squaredTensor = tf.square(target - gen_output)
    l2_loss = tf.reduce_mean(squaredTensor)
    ssim_loss = CalculateDSSIM(target, gen_output, kernel_size=7, k1=0.01, k2=0.03, max_value=1.0)
    total_gen_loss = (R_LAMBDA_DISC * cheated_disc_loss) + (R_LAMBDA_MSE * l2_loss) + (R_LAMBDA_SSIM * ssim_loss)
    return total_gen_loss, cheated_disc_loss, l2_loss, ssim_loss

def rec_discriminator_loss(disc_real_output, disc_generated_output):
    squaredTensor = tf.square(tf.ones_like(disc_real_output) - disc_real_output)
    real_recognition_loss = tf.reduce_mean(squaredTensor)
    real_accuracy = 100 - (tf.reduce_mean(tf.abs(tf.ones_like(disc_real_output) - disc_real_output)) * 100)
    
    squaredTensor = tf.square(tf.zeros_like(disc_generated_output) - disc_generated_output)
    generated_recognition_loss = tf.reduce_mean(squaredTensor)
    generated_accuracy = 100 - (tf.reduce_mean(tf.abs(tf.zeros_like(disc_generated_output) - disc_generated_output)) * 100)
    
    total_disc_loss = real_recognition_loss + generated_recognition_loss

    return total_disc_loss, real_recognition_loss, generated_recognition_loss, real_accuracy, generated_accuracy

# Network initialization
projection_matrix = make_projection_matrix(parameters)
lrtsg, lrtsd, lrtrg, lrtrd = 2e-4, 2e-4, 2e-4, 2e-4
sino_generator_optimizer = tf.keras.optimizers.Adam(lrtsg, beta_1=0.9, beta_2=0.999)
sino_discriminator_optimizer = tf.keras.optimizers.Adam(lrtsd, beta_1=0.9, beta_2=0.999)
rec_generator_optimizer = tf.keras.optimizers.Adam(lrtrg, beta_1=0.9, beta_2=0.999)
rec_discriminator_optimizer = tf.keras.optimizers.Adam(lrtrd, beta_1=0.9, beta_2=0.999)


sino_generator = sinoGenerator()
sino_discriminator = sinoDiscriminator()
rec_generator = recGenerator()
rec_discriminator = recDiscriminator()


checkpoint_dir = "./training_dual_gan/" + name
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_")
if not os.path.exists(checkpoint_prefix):
        os.makedirs(checkpoint_prefix)
checkpoint = tf.train.Checkpoint(sino_generator_optimizer = sino_generator_optimizer,
                                 sino_discriminator_optimizer = sino_discriminator_optimizer,
                                 sino_generator = sino_generator,
                                 sino_discriminator = sino_discriminator,
                                 rec_generator_optimizer = rec_generator_optimizer,
                                 rec_discriminator_optimizer = rec_discriminator_optimizer,
                                 rec_generator = rec_generator,
                                 rec_discriminator = rec_discriminator)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=10)

currentTime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir="logs/"
if not os.path.exists(log_dir):
        os.makedirs(log_dir)
log_file = log_dir + name + "-" + currentTime
summary_writer = tf.summary.create_file_writer(log_file)

def generate_images(model_sino, model_rec, test_input, tar_sino, tar_rec):
    sino_generator.trainable = False
    sino_discriminator.trainable = False
    rec_generator.trainable = False
    rec_discriminator.trainable = False
    prediction_sino = model_sino(test_input, training = True)
    prediction_rec = model_rec(prediction_sino, training = True)
    plt.figure(figsize=(15, 15))
    display_list = [test_input[0], tar_sino[0], prediction_sino[0], tar_rec[0], prediction_rec[0]]
    title = ['Sparse sinogram', 'GT sinogram', 'Predicted sinogram', 'GT reconstruction', 'Predicted reconstruction']
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.title(title[i])
        plt.imshow(np.squeeze(display_list[i]))
        plt.axis('off')
    plt.show()

# ONLY SINO GENERATOR
@tf.function
def train_step_1(input_image, target, step):
    sino_generator.trainable = True
    sino_discriminator.trainable = False
    rec_generator.trainable = False
    rec_discriminator.trainable = False
    
    with tf.GradientTape() as sino_gen_tape:
        gen_output = sino_generator(input_image, training=True)
        gen_total_loss, gen_l2_loss, gen_ssim_loss = sino_generator_loss(gen_output, target)

    sino_generator_gradients = sino_gen_tape.gradient(gen_total_loss, sino_generator.trainable_variables)
    sino_generator_optimizer.apply_gradients(zip(sino_generator_gradients, sino_generator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('sino_gen_total_loss', gen_total_loss, step=step)
        tf.summary.scalar('sino_gen_cheated_disc_loss', 0, step=step)
        tf.summary.scalar('sino_gen_l2_loss', gen_l2_loss*S_LAMBDA_MSE, step=step)
        tf.summary.scalar('sino_gen_ssim_loss', gen_ssim_loss*S_LAMBDA_SSIM, step=step)
        tf.summary.scalar('sino_disc_loss', 0, step=step)
        tf.summary.scalar('sino_disc_real_loss', 0, step=step)
        tf.summary.scalar('sino_disc_generated_loss', 0, step=step)
        tf.summary.scalar('sino_disc_real_accuracy', 0, step=step)
        tf.summary.scalar('sino_disc_generated_accuracy', 0, step=step)
        tf.summary.scalar('rec_gen_total_loss', 0, step=step)
        tf.summary.scalar('rec_gen_cheated_disc_loss', 0, step=step)
        tf.summary.scalar('rec_gen_l2_loss', 0, step=step)
        tf.summary.scalar('rec_gen_ssim_loss', 0, step=step)
        tf.summary.scalar('rec_disc_loss', 0, step=step)
        tf.summary.scalar('rec_disc_real_loss', 0, step=step)
        tf.summary.scalar('rec_disc_generated_loss', 0, step=step)
        tf.summary.scalar('rec_disc_real_accuracy', 0, step=step)
        tf.summary.scalar('rec_disc_generated_accuracy', 0, step=step)

# ONLY SINO GENERATOR AND DISCRIMINATOR
@tf.function
def train_step_2(input_image, target, step):
    sino_generator.trainable = True
    sino_discriminator.trainable = True
    rec_generator.trainable = False
    rec_discriminator.trainable = False
    
    with tf.GradientTape() as sino_gen_tape, tf.GradientTape() as sino_disc_tape:
        gen_output = sino_generator(input_image, training=True)
        disc_real_output = sino_discriminator(target, training=True)
        disc_generated_output = sino_discriminator(gen_output, training=True)
        gen_total_loss, gen_cheated_disc_loss, gen_l2_loss, gen_ssim_loss = sino_generator_loss2(disc_generated_output, gen_output, target)
        disc_loss, disc_real_loss, disc_generated_loss, disc_real_acc, disc_gen_acc = sino_discriminator_loss(disc_real_output, disc_generated_output)

    sino_generator_gradients = sino_gen_tape.gradient(gen_total_loss, sino_generator.trainable_variables)
    sino_discriminator_gradients = sino_disc_tape.gradient(disc_loss, sino_discriminator.trainable_variables)
    sino_generator_optimizer.apply_gradients(zip(sino_generator_gradients, sino_generator.trainable_variables))
    sino_discriminator_optimizer.apply_gradients(zip(sino_discriminator_gradients, sino_discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('sino_gen_total_loss', gen_total_loss, step=step)
        tf.summary.scalar('sino_gen_cheated_disc_loss', gen_cheated_disc_loss*S_LAMBDA_DISC, step=step)
        tf.summary.scalar('sino_gen_l2_loss', gen_l2_loss*S_LAMBDA_MSE, step=step)
        tf.summary.scalar('sino_gen_ssim_loss', gen_ssim_loss*S_LAMBDA_SSIM, step=step)
        tf.summary.scalar('sino_disc_loss', disc_loss, step=step)
        tf.summary.scalar('sino_disc_real_loss', disc_real_loss, step=step)
        tf.summary.scalar('sino_disc_generated_loss', disc_generated_loss, step=step)
        tf.summary.scalar('sino_disc_real_accuracy', disc_real_acc, step=step)
        tf.summary.scalar('sino_disc_generated_accuracy', disc_gen_acc, step=step)
        tf.summary.scalar('rec_gen_total_loss', 0, step=step)
        tf.summary.scalar('rec_gen_cheated_disc_loss', 0, step=step)
        tf.summary.scalar('rec_gen_l2_loss', 0, step=step)
        tf.summary.scalar('rec_gen_ssim_loss', 0, step=step)
        tf.summary.scalar('rec_disc_loss', 0, step=step)
        tf.summary.scalar('rec_disc_real_loss', 0, step=step)
        tf.summary.scalar('rec_disc_generated_loss', 0, step=step)
        tf.summary.scalar('rec_disc_real_accuracy', 0, step=step)
        tf.summary.scalar('rec_disc_generated_accuracy', 0, step=step)

# ONLY REC GEN (TRAINING ON GT)        
@tf.function
def train_step_3(input_image, target, step):
    sino_generator.trainable = False
    sino_discriminator.trainable = False
    rec_generator.trainable = True
    rec_discriminator.trainable = False
    
    with tf.GradientTape() as rec_gen_tape:
        gen_output = rec_generator(input_image, training=True)
        gen_total_loss, gen_l2_loss, gen_ssim_loss = rec_generator_loss(gen_output, target)

    rec_generator_gradients = rec_gen_tape.gradient(gen_total_loss, rec_generator.trainable_variables)
    rec_generator_optimizer.apply_gradients(zip(rec_generator_gradients, rec_generator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('sino_gen_total_loss', 0, step=step)
        tf.summary.scalar('sino_gen_cheated_disc_loss', 0, step=step)
        tf.summary.scalar('sino_gen_l2_loss', 0, step=step)
        tf.summary.scalar('sino_gen_ssim_loss', 0, step=step)
        tf.summary.scalar('sino_disc_loss', 0, step=step)
        tf.summary.scalar('sino_disc_real_loss', 0, step=step)
        tf.summary.scalar('sino_disc_generated_loss', 0, step=step)
        tf.summary.scalar('sino_disc_real_accuracy', 0, step=step)
        tf.summary.scalar('sino_disc_generated_accuracy', 0, step=step)
        tf.summary.scalar('rec_gen_total_loss', gen_total_loss, step=step)
        tf.summary.scalar('rec_gen_cheated_disc_loss', 0, step=step)
        tf.summary.scalar('rec_gen_l2_loss', gen_l2_loss*R_LAMBDA_MSE, step=step)
        tf.summary.scalar('rec_gen_ssim_loss', gen_ssim_loss*R_LAMBDA_SSIM, step=step)
        tf.summary.scalar('rec_disc_loss', 0, step=step)
        tf.summary.scalar('rec_disc_real_loss', 0, step=step)
        tf.summary.scalar('rec_disc_generated_loss', 0, step=step)
        tf.summary.scalar('rec_disc_real_accuracy', 0, step=step)
        tf.summary.scalar('rec_disc_generated_accuracy', 0, step=step)        

# ONLY RECONSTRUCTION GENERATOR AND DISCRIMINATOR (TRAINING ON GT)
@tf.function
def train_step_4(input_image, target, step):
    sino_generator.trainable = False
    sino_discriminator.trainable = False
    rec_generator.trainable = True
    rec_discriminator.trainable = True
    
    with tf.GradientTape() as rec_gen_tape, tf.GradientTape() as rec_disc_tape:
        gen_output = rec_generator(input_image, training=True)
        disc_real_output = rec_discriminator(target, training=True)
        disc_generated_output = rec_discriminator(gen_output, training=True)
        gen_total_loss, gen_cheated_disc_loss, gen_l2_loss, gen_ssim_loss = rec_generator_loss2(disc_generated_output, gen_output, target)
        disc_loss, disc_real_loss, disc_generated_loss, disc_real_acc, disc_gen_acc = rec_discriminator_loss(disc_real_output, disc_generated_output)
    
    rec_generator_gradients = rec_gen_tape.gradient(gen_total_loss, rec_generator.trainable_variables)
    rec_discriminator_gradients = rec_disc_tape.gradient(disc_loss, rec_discriminator.trainable_variables)
    rec_generator_optimizer.apply_gradients(zip(rec_generator_gradients, rec_generator.trainable_variables))
    rec_discriminator_optimizer.apply_gradients(zip(rec_discriminator_gradients, rec_discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('sino_gen_total_loss', 0, step=step)
        tf.summary.scalar('sino_gen_cheated_disc_loss', 0, step=step)
        tf.summary.scalar('sino_gen_l2_loss', 0, step=step)
        tf.summary.scalar('sino_gen_ssim_loss', 0, step=step)
        tf.summary.scalar('sino_disc_loss', 0, step=step)
        tf.summary.scalar('sino_disc_real_loss', 0, step=step)
        tf.summary.scalar('sino_disc_generated_loss', 0, step=step)
        tf.summary.scalar('sino_disc_real_accuracy', 0, step=step)
        tf.summary.scalar('sino_disc_generated_accuracy', 0, step=step)
        tf.summary.scalar('rec_gen_total_loss', gen_total_loss, step=step)
        tf.summary.scalar('rec_gen_cheated_disc_loss', gen_cheated_disc_loss*R_LAMBDA_DISC, step=step)
        tf.summary.scalar('rec_gen_l2_loss', gen_l2_loss*R_LAMBDA_MSE, step=step)
        tf.summary.scalar('rec_gen_ssim_loss', gen_ssim_loss*R_LAMBDA_SSIM, step=step)
        tf.summary.scalar('rec_disc_loss', disc_loss, step=step)
        tf.summary.scalar('rec_disc_real_loss', disc_real_loss, step=step)
        tf.summary.scalar('rec_disc_generated_loss', disc_generated_loss, step=step)
        tf.summary.scalar('rec_disc_real_accuracy', disc_real_acc, step=step)
        tf.summary.scalar('rec_disc_generated_accuracy', disc_gen_acc, step=step)  
        
# SINO GEN AND DISC AND REC GEN AND DISC. 
@tf.function
def train_step_5(input_image, target1, target2, step):
    sino_generator.trainable = True
    sino_discriminator.trainable = True
    rec_generator.trainable = True
    rec_discriminator.trainable = True
    
    with tf.GradientTape() as sino_gen_tape, tf.GradientTape() as sino_disc_tape, tf.GradientTape() as rec_gen_tape, tf.GradientTape() as rec_disc_tape: #, tf.GradientTape() as sino_rec_gen_tape:
        gen_output1 = sino_generator(input_image, training=True)
        disc_real_output1 = sino_discriminator(target1, training=True)
        disc_generated_output1 = sino_discriminator(gen_output1, training=True)
        gen_total_loss1, gen_cheated_disc_loss1, gen_l2_loss1, gen_ssim_loss1 = sino_generator_loss2(disc_generated_output1, gen_output1, target1)
        disc_loss1, disc_real_loss1, disc_generated_loss1, disc_real_acc1, disc_gen_acc1 = sino_discriminator_loss(disc_real_output1, disc_generated_output1)
  
        gen_output2 = rec_generator(gen_output1, training=True)
        disc_real_output2 = rec_discriminator(target2, training=True)
        disc_generated_output2 = rec_discriminator(gen_output2, training=True)
        gen_total_loss2, gen_cheated_disc_loss2, gen_l2_loss2, gen_ssim_loss2 = rec_generator_loss2(disc_generated_output2, gen_output2, target2)
        disc_loss2, disc_real_loss2, disc_generated_loss2, disc_real_acc2, disc_gen_acc2 = rec_discriminator_loss(disc_real_output2, disc_generated_output2)
        
        ALPHA = 8
        sino_loss_for_back = (ALPHA * gen_total_loss1) + gen_total_loss2
    
    
    sino_generator_gradients = sino_gen_tape.gradient(sino_loss_for_back, sino_generator.trainable_variables)
    sino_generator_optimizer.apply_gradients(zip(sino_generator_gradients, sino_generator.trainable_variables))
    sino_discriminator_gradients = sino_disc_tape.gradient(disc_loss1, sino_discriminator.trainable_variables)
    sino_discriminator_optimizer.apply_gradients(zip(sino_discriminator_gradients, sino_discriminator.trainable_variables))
    
    rec_generator_gradients = rec_gen_tape.gradient(gen_total_loss2, rec_generator.trainable_variables)
    rec_generator_optimizer.apply_gradients(zip(rec_generator_gradients, rec_generator.trainable_variables))
    rec_discriminator_gradients = rec_disc_tape.gradient(disc_loss2, rec_discriminator.trainable_variables)
    rec_discriminator_optimizer.apply_gradients(zip(rec_discriminator_gradients, rec_discriminator.trainable_variables))

    
    with summary_writer.as_default():
        tf.summary.scalar('sino_gen_total_loss', gen_total_loss1, step=step)
        tf.summary.scalar('sino_gen_cheated_disc_loss', gen_cheated_disc_loss1*S_LAMBDA_DISC, step=step)
        tf.summary.scalar('sino_gen_l2_loss', gen_l2_loss1*S_LAMBDA_MSE, step=step)
        tf.summary.scalar('sino_gen_ssim_loss', gen_ssim_loss1*S_LAMBDA_SSIM, step=step)
        tf.summary.scalar('sino_disc_loss', disc_loss1, step=step)
        tf.summary.scalar('sino_disc_real_loss', disc_real_loss1, step=step)
        tf.summary.scalar('sino_disc_generated_loss', disc_generated_loss1, step=step)
        tf.summary.scalar('sino_disc_real_accuracy', disc_real_acc1, step=step)
        tf.summary.scalar('sino_disc_generated_accuracy', disc_gen_acc1, step=step)
        tf.summary.scalar('rec_gen_total_loss', gen_total_loss2, step=step)
        tf.summary.scalar('rec_gen_cheated_disc_loss', gen_cheated_disc_loss2*R_LAMBDA_DISC, step=step)
        tf.summary.scalar('rec_gen_l2_loss', gen_l2_loss2*R_LAMBDA_MSE, step=step)
        tf.summary.scalar('rec_gen_ssim_loss', gen_ssim_loss2*R_LAMBDA_SSIM, step=step)
        tf.summary.scalar('rec_disc_loss', disc_loss2, step=step)
        tf.summary.scalar('rec_disc_real_loss', disc_real_loss2, step=step)
        tf.summary.scalar('rec_disc_generated_loss', disc_generated_loss2, step=step)
        tf.summary.scalar('rec_disc_real_accuracy', disc_real_acc2, step=step)
        tf.summary.scalar('rec_disc_generated_accuracy', disc_gen_acc2, step=step)

        
def fit(train_ds, test_ds, steps):
    example_input, example_target_sino, example_target_rec = next(iter(test_ds.take(1)))
    start = time.time()
    for step, (input_image, target_sino, target_rec) in train_ds.repeat().take(steps).enumerate():
        if (step) % 1000 == 0:
            display.clear_output(wait=True)
            if step != 0:
                print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec.\n')
            start = time.time()
            generate_images(sino_generator, rec_generator, example_input, example_target_sino, example_target_rec)
            print(f"Step: {step//1000}k")
            
        # Training part 
        
        
        if step.numpy()<15000:
            train_step_1(input_image, target_sino, step)
        elif step.numpy()>=15000 and step.numpy()<50000:
            train_step_2(input_image, target_sino, step)
        elif step.numpy()>=50000 and step.numpy()<65000:
            train_step_3(target_sino, target_rec, step)
        elif (step.numpy()>=65000 and step.numpy()<100000):
            train_step_4(target_sino, target_rec, step)
        else:
            train_step_5(input_image, target_sino, target_rec, step)    
            

            
        # Printing
        if (step+1) % 10 == 0:
            print('.', end='', flush=True)
        # Save (checkpoint) the model every 1k steps
        if (step + 1) % 1000 == 0:
            manager.save()
            if (step+1)==15000:
                if not os.path.exists(checkpoint_dir + "/ckpt_/saved_step1/"):
                    os.makedirs(checkpoint_dir + "/ckpt_/saved_step1/")
                checkpoint.save(checkpoint_dir + "/ckpt_/saved_step1/s")
            elif (step+1)==50000:
                if not os.path.exists(checkpoint_dir + "/ckpt_/saved_step2/"):
                    os.makedirs(checkpoint_dir + "/ckpt_/saved_step2/")
                checkpoint.save(checkpoint_dir + "/ckpt_/saved_step2/s")
            elif (step+1)==65000:
                if not os.path.exists(checkpoint_dir + "/ckpt_/saved_step3/"):
                    os.makedirs(checkpoint_dir + "/ckpt_/saved_step3/")
                checkpoint.save(checkpoint_dir + "/ckpt_/saved_step3/s") 
            elif (step+1)==100000:
                if not os.path.exists(checkpoint_dir + "/ckpt_/saved_step4/"):
                    os.makedirs(checkpoint_dir + "/ckpt_/saved_step4/")
                checkpoint.save(checkpoint_dir + "/ckpt_/saved_step4/s")
            elif (step+1)==130000:
                if not os.path.exists(checkpoint_dir + "/ckpt_/saved_step5/"):
                    os.makedirs(checkpoint_dir + "/ckpt_/saved_step5/")
                checkpoint.save(checkpoint_dir + "/ckpt_/saved_step5/s")

STEPS = 130000
fit(train_dataset, test_dataset, steps=STEPS)                