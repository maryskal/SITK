import os
import numpy as np
import tensorflow as tf
from sklearn import model_selection
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import models
from tensorflow.keras import layers
import image_funct as im
import extra_functions as ex
import logs

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

path = '/home/mr1142/Documents/Data/rx_diseases'
name = ''
batch = 8
epoch = 100

pix = 256

sano, enfermo = [os.listdir(os.path.join(path, dir)) for dir in os.listdir(path)]

mask_model = ex.charge_model('unet_1chan_5_600', True)

tensor = np.zeros((len(sano)+len(enfermo), pix,pix,1))
Y = np.zeros((len(sano)+len(enfermo)))
for i in range(len(sano)):
    tensor[i,...] = im.norm_clahe(im.apply_mask(im.read_img(path, 'sano', sano[i], pix), mask_model))

for i in range(len(enfermo)):
    j = len(sano)+i
    tensor[j,...] = im.norm_clahe(im.apply_mask(im.read_img(path, 'enfermo', enfermo[i], pix), mask_model))
    Y[j] = 1



X, X_val, Y, Y_val = model_selection.train_test_split(tensor, Y, test_size=0.2, 
                                                        random_state=42, shuffle=True, stratify=Y)


# # Modelo

# - Añado una capa inicial para poder entregarle el input de 3 canales. 
# - Añado 1 convolución con max pooling y una segunda convolución cno globalMaxPooling para ajustar mejor
# - Añado Dropout para evitar overfiting
# - Añado la capa final con 1 neurona de salida


input_shape = (pix,pix,3)
conv_base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)

model = models.Sequential()
model.add(layers.Conv2D(3,3,padding="same", input_shape=(pix,pix,1), activation='elu', name = 'conv_inicial'))
model.add(conv_base)
model.add(layers.Conv2D(3,32, padding='same', input_shape=(8,8,1280), activation='selu', name = 'conv_posterior'))
model.add(layers.MaxPool2D(pool_size = (2,2), padding='same', name = 'first_pooling'))
model.add(layers.Conv2D(3,64, padding='same', input_shape=(4,4,1280), activation='selu', name = 'last_convolution'))
model.add(layers.GlobalMaxPooling2D(name="general_max_pooling"))
model.add(layers.Dropout(0.2, name="dropout_out"))
model.add(layers.Dense(1, activation="sigmoid", name="fc_out"))


# # Compilación y entrenamiento

# ## Callback

callb = [logs.tensorboard('image_class' + name), logs.early_stop(7)]


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



model.compile(optimizer=opt, loss = loss , metrics = met)


history = model.fit(X,Y,
                    batch_size = batch,
                    epochs = epoch,
			        callbacks = callb,
                    shuffle = True,
                    validation_split = 0.2)

model.save('/home/mr1142/Documents/Data/models/imagenes.h5')
