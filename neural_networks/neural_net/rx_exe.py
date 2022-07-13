import os
import pandas as pd
import numpy as np
import PIL
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import cv2

from sklearn import model_selection
from PIL import Image
from skimage import exposure

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import models
from tensorflow.keras import layers
import tensorflow_addons as tfa

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


path = '/home/maryskal/Downloads/archive/vinbigdata'
path = '/home/mr1142/Documents/Data/vinbigdata'


df = pd.read_csv(os.path.join(path, 'train.csv'))

values = ['Consolidation']
df['enfermo'] = 0
index = [i for i, clase in enumerate(df.class_name) if clase in values]
df['enfermo'][index] = 1



# Divido el dataset en dos y reduzco el de no neumonía


enfermo = df[df.enfermo == 1].reset_index(drop=True)
sano = df[df.class_name == 'No finding'].reset_index(drop=True)


# Para reducir elijo un numero de valores aleatorios que serán los índices


selection = np.random.randint(0, len(sano), len(enfermo))
sano = sano.iloc[selection,:]

# Genero el nuevo dataframe juntando

df = pd.concat([enfermo, sano]).reset_index(drop=True)


# Reordeno aleatoriamente

df = df.sample(frac=1)


# ## Imagenes

from tensorflow.keras import backend as K
from tensorflow import keras

pixels = 512


# ### Cargamos la máscara

def dice_coef(y_true, y_pred, smooth=100):        
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

mask_model = keras.models.load_model('/home/mr1142/Documents/Data/models/unet_1chan_1.h5', 
                                     custom_objects={"dice_coef_loss": dice_coef_loss, "dice_coef": dice_coef})


def read_img(path, folder, img):
    img = cv2.imread(os.path.join(path, folder, img))
    return img


def normalize(img):
    return (img - np.mean(img))/ np.std(img)


def prepare_img(img, pix):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (pix, pix))
    img = np.expand_dims(img, axis=-1)
    img = normalize(img)
    return img


def prepare_img_clahe(img, pix):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE()
    img = clahe.apply(img)
    img = cv2.resize(img, (pix, pix))
    img = np.expand_dims(img, axis=-1)
    img = normalize(img)
    return img


def apply_mask(img):
    pix = img.shape[1]
    img_2 = prepare_img(img, 256)
    mask = mask_model.predict(img_2[np.newaxis,...])[0,...]
    mask = cv2.resize(mask, (pix, pix))
    img[mask!=1]=0
    return img



tensor = np.zeros((len(df.image_id), pixels, pixels, 1))
for i in range(tensor.shape[0]):
    img = apply_mask(read_img(path, 'train', df.image_id[i] +'.png'))
    tensor[i,...] = prepare_img_clahe(img, 512)



# ## X e Y

Y = np.array(df['enfermo'])
X = tensor

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, 
                                                                    random_state=42, shuffle=True, stratify=Y)

# # Modelo

# - Añado una capa inicial para poder entregarle el input de 3 canales. 
# - Añado 1 convolución con max pooling y una segunda convolución cno globalMaxPooling para ajustar mejor
# - Añado Dropout para evitar overfiting
# - Añado la capa final con 1 neurona de salida

# In[144]:


input_shape = (pixels,pixels,3)
conv_base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)

model = models.Sequential()
model.add(layers.Conv2D(3,3,padding="same", input_shape=(pixels,pixels,1), activation='elu', name = 'conv_inicial'))
model.add(conv_base)
model.add(layers.Conv2D(3,32, padding='same', input_shape=(8,8,1280), activation='selu', name = 'conv_posterior'))
model.add(layers.MaxPool2D(pool_size = (2,2), padding='same', name = 'first_pooling'))
model.add(layers.Conv2D(3,64, padding='same', input_shape=(4,4,1280), activation='selu', name = 'last_convolution'))
model.add(layers.GlobalMaxPooling2D(name="general_max_pooling"))
model.add(layers.Dropout(0.2, name="dropout_out"))
model.add(layers.Dense(1, activation="sigmoid", name="fc_out"))


# # Compilación y entrenamiento

# ## Callback

log_dir = "/home/mr1142/Documents/Data/logs/fit/image" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                      update_freq='batch',
                                                       histogram_freq=1)
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", restore_best_weights=True, patience = 10)

# ## Entrenamiento

# Voy a entrenar desde la capa 100
fine_tune_at = 100

for layer in conv_base.layers[:fine_tune_at]:
    layer.trainable = False

# ### Parámetros

lr = 1e-4

loss = tf.keras.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(learning_rate = lr)
met = ['BinaryAccuracy', 'AUC']


batch = 8
epoch = 100

model.compile(optimizer=opt, loss = loss , metrics = met)


history = model.fit(X_train,Y_train,
                    batch_size = batch,
                    epochs = epoch,
			callbacks = [tensorboard_callback, early_stop]
                    shuffle = True,
                    validation_split = 0.2)

unet_model.save('/home/mr1142/Documents/Data/models/images_neumo.h5')
