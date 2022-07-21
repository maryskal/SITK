import argparse
import os
import logs
import numpy as np
import tensorflow as tf
import logging_function as log

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
    import unet_doble_loss as un
    import image_funct as im
    import extra_functions as ex


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

    unet_model = un.build_unet_model(pixels)
    unet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-4),
                        loss=un.MyLoss,
                        metrics = [ex.dice_coef_loss, un.loss_mask])

    if callbacks:
        callb = [logs.tensorboard('uloss_' + name), logs.early_stop(7)]
    else:
        callb = []

    history = unet_model.fit(images,masks,
                         batch_size = batch,
                         epochs = epoch,
                         callbacks = callb,
                         shuffle = True,
                         validation_split=0.2,
                         verbose = 1)

    unet_model.save('/home/mr1142/Documents/Data/models/unet_custom_loss' + name + '.h5')