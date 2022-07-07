import image_funct as im
import unet_funct as un
import tensorflow as tf
import logs
import os

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    path = '/home/mr1142/Documents/Data/seg_prueba'
    batch = 8
    epoch = 100
    pixels = 256

    masks_name = os.listdir(os.path.join(path, 'mascara'))

    masks = im.create_tensor(path, 'mascara', masks_name, pixels, im.normalize)
    images = im.create_tensor(path, 'images', masks_name, pixels, im.binarize)

    images, masks = im.double_tensor(images,masks)

    unet_model = un.build_unet_model(pixels)

    callb = [logs.tensorboard('U_net'), logs.weights('U_net'), logs.early_stop(7)]

    unet_model.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy", "mean_squared_error"])


    history = unet_model.fit(images,masks,
                            batch_size = batch,
                            epochs = epoch,
                            callbacks = callb,
                            shuffle = True,
                            validation_split = 0.2)