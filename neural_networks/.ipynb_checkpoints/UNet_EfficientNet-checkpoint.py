
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import tensorflow_datasets as tfds
from tensorflow.keras.applications import EfficientNetB4


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import logging
import sys

# Configura el logging
log_format = '[%(process)d]\t%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(format=log_format, level=logging.INFO, datefmt="%H:%M:%S",
                    handlers=[logging.StreamHandler(sys.stdout)])

# # Modelo
# https://www.nature.com/articles/s41598-022-12743-y/figures/2


def encoder_subblock(efNet_conection_output):
    x = layers.Conv2D(512,3, padding = 'same')(efNet_conection_output)
    x = residual_block(x)
    x = residual_block(x)
    x = layers.LeakyReLU()(x)
    return x


def efNet_conection(efNet_output):
    x1 = layers.LeakyReLU()(efNet_output)
    x2 = layers.MaxPool2D((2,2), padding = 'same')(x1)
    x3 = layers.Dropout(0.3)(x2)
    return x3


def decoder_subblock(encoder_output, prev_layer):
    # https://www.nature.com/articles/s41598-022-12743-y/figures/3
    unification = layers.concatenate([encoder_output, prev_layer])
    x1 = layers.Dropout(0.3)(unification)
    x2 = layers.Conv2D(3,3,padding = 'same')(x1)
    x3 = residual_block(x2)
    x3 = residual_block(x3)
    x4 = layers.LeakyReLU()(x3)
    return x4


def residual_block(prev_layer):
    path_1 = layers.LeakyReLU()(prev_layer)
    path_1 = layers.BatchNormalization()(path_1)
    path_1 = layers.Conv2D(3,3,padding = 'same')(path_1)
    path_1 = layers.BatchNormalization()(path_1)
    path_1 = layers.LeakyReLU()(path_1)
    path_1 = layers.Conv2D(3,3,padding = 'same')(path_1)
    path_1 = layers.BatchNormalization()(path_1)
    path_2 = layers.BatchNormalization()(prev_layer)
    return layers.concatenate([path_1, path_2])

    
def up_sampling(prev_layer, deep, kernel):
    x = layers.Conv2DTranspose(deep, kernel)(prev_layer)
    return x


def last_up_sampling(prev_layer):
    x = layers.Conv2D(1,3, padding='same', activation='sigmoid')(prev_layer)
    return x



def build_unet_model():
     # inputs
    inputs = layers.Input(shape=(256,256,3))
    
    # adaptation = layers.Conv2D(3, 3, padding="same", activation = "elu")(inputs)
        
    #EfficientNet
    efficienNet = EfficientNetB4(weights="imagenet",
                                 include_top=False,
                                 input_shape=(256,256,3),
                                 input_tensor=inputs)
    
    efficienNet.trainable = False
    
    # ENCONDING
    # (128,128,144)
    dw1 = efficienNet.layers[31].output
    # (64,64,192)
    dw2 = efficienNet.layers[90].output
    # (32,32,336)
    dw3 = efficienNet.layers[149].output
    # (16,16,960)
    dw4 = efficienNet.layers[326].output

    # (8,8,960)
    middle = efNet_conection(dw4)
    # Encoder sub block (no la he añadido porque me da error)
    middle = encoder_subblock(middle)
    
    # DECODING
    # (16,16,960)
    uc1 = up_sampling(middle, dw4.shape[3],9)
    up1 = decoder_subblock(dw4, uc1)
    
    # (32,32,336)
    uc2 = up_sampling(up1, dw3.shape[3],17)
    up2 = decoder_subblock(dw3, uc2)
    
    # (64,64,192)
    uc3 = up_sampling(up2, dw2.shape[3],33)
    up3 = decoder_subblock(dw2, uc3)
    
    # (128,128,144)
    uc4 = up_sampling(up3,dw3.shape[3],65)
    up4 = decoder_subblock(dw1, uc4)

    uc5 = up_sampling(up4,16,129)

    # outputs
    outputs = last_up_sampling(uc5)

    # unet model with Keras Functional API
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

    return unet_model


unet_model = build_unet_model()
logging.info('BUILDED')
unet_model.summary()

model = tf.keras.models.Sequential()
model.add(layers.Conv2D(3,3,padding="same", input_shape=(256,256,1), activation='elu', name = 'conv_inicial'))
model.add(unet_model)
logging.info('BUILDED Model 2')
model.summary()

# **Capas a entrenar**
# 
# Vamos a entrenar la primera capa y a partir de la capa 327 de unet-model, que es la ultima del modelo preentrenado

# In[11]:


fine_tune_at = 327

for layer in unet_model.layers[1:327]:
    layer.trainable = False




# # Datos
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk


# # Datos

path = '/home/mr1142/Documents/Data/seg_prueba'

masks_name = os.listdir(os.path.join(path, 'mascara'))
pixels = 256

def read_img(path, folder, img):
    img = cv2.imread(os.path.join(path, folder, img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (pixels, pixels))
    img = np.expand_dims(img, axis=-1)
    return img

def normalize(img):
    return (img - np.mean(img))/ np.std(img)


def binarize(img):
    img[img>0] = 1
    return img


masks = np.zeros((len(masks_name), pixels,pixels,1))
print(masks.shape)
for i in range(len(masks_name)):
    masks[i, ...] = binarize(read_img(path, 'mascara', masks_name[i]))


images = np.zeros((len(masks_name), pixels,pixels,1))
for i in range(len(masks_name)):
    images[i, ...] = normalize(read_img(path, 'images', masks_name[i]))



import imutils

def augment(input_image, input_mask):
    r = np.random.randint(-60,60)
    # Random flipping of the image and mask
    input_image = np.expand_dims(imutils.rotate(input_image, angle=r),  axis=-1)
    input_mask = np.expand_dims(imutils.rotate(input_mask, angle=r), axis=-1)
    return input_image, input_mask


# Nuevas imagenes con rotacion random

new_img = np.zeros((len(masks_name), pixels,pixels,1))
new_mask = np.zeros((len(masks_name), pixels,pixels,1))
for i in range(len(masks_name)):
    img, mask = augment(images[i], masks[i])
    new_img[i, ...] = img
    new_mask[i,...] = mask


images = np.concatenate((new_img, images), axis = 0)
masks = np.concatenate((new_mask, masks), axis = 0)


# # Entrenamiento
model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])


batch = 1
epoch = 500


history = model.fit(images,masks,
                    batch_size = batch,
                    epochs = epoch,
                    shuffle = True,
                    validation_split = 0.2)





