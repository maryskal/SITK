import argparse
import os
import logs
import numpy as np
import tensorflow as tf
import image_funct as im
import logging_function as log
import unet_doble_loss as un


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--device',
                        help="GPU device",
                        type=str,
                        default=2)
    parser.add_argument('-p',
                        '--path',
                        help="Images path",
                        type=str,
                        default='/home/mr1142/Documents/Data/segmentation')
    parser.add_argument('-e',
                        '--epochs',
                        type=int,
                        default=200,
                        help="Number of Epochs")
    parser.add_argument('-pi',
                        '--pixels',
                        type=int,
                        default=256,
                        help="Pixels for image")
    parser.add_argument('-bs',
                        '--batch_size',
                        type=int,
                        default=8,
                        help="Batch size")
    parser.add_argument('-c',
                        '--callbacks',
                        type=bool,
                        default=True,
                        help="Callbacks")
    parser.add_argument('-n',
                        '--name',
                        type=str,
                        default='new',
                        help="name of the model")                               
    parser.add_argument('-cl',
                        '--clahe',
                        type=bool,
                        default=False,
                        help="apply clahe to rx") 

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    path = str(args.path)
    batch = args.batch_size
    epoch = args.epochs
    pixels = args.pixels
    callbacks = args.callbacks
    name = args.name
    clahe = args.clahe

    masks_name = os.listdir(os.path.join(path, 'mascara'))

    masks = im.create_tensor(path, 'mascara', masks_name, im.binarize, pixels)
    if clahe:
        images = im.create_tensor(path, 'images', masks_name, im.norm_clahe, pixels)
    else:
        images = im.create_tensor(path, 'images', masks_name, im.normalize, pixels)
    log.information('Unet', 'Imagenes cargadas')

    images, masks = im.double_tensor(images,masks)

    sub_mask = un.charge_mask()

    unet_model = un.build_unet_model(pixels, sub_mask)
    unet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-4),
                        loss=None)

    if callbacks:
        callb = [logs.tensorboard('unet_custom_loss_' + name), logs.early_stop(7)]
    else:
        callb = []

    history = unet_model.fit([images,masks],
                         batch_size = batch,
                         epochs = epoch,
                         callbacks = callb,
                         shuffle = True,
                         validation_split=0.2,
                         verbose = 1)

    unet_model.save('/home/mr1142/Documents/Data/models/unet_custom_loss' + name + '.h5')