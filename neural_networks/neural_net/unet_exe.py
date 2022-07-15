from gc import callbacks
import image_funct as im
import unet_funct as un
import extra_functions as ex
import logging_function as log
import tensorflow as tf
import logs
import argparse
import os


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
    parser.add_argument('-ch',
                        '--chanels',
                        type=int,
                        default=3,
                        help="Chanels")
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
    output_chanels = args.chanels
    name = args.name
    clahe = args.clahe

    masks_name = os.listdir(os.path.join(path, 'mascara'))

    masks = im.create_tensor(path, 'mascara', masks_name, im.binarize, pixels)
    if clahe:
        images = im.create_tensor(path, 'images', masks_name, im.norm_clahe, pixels)
    else:
        images = im.create_tensor(path, 'images', masks_name, im.normalize, pixels)

    images, masks = im.double_tensor(images,masks)
    
    log.information('Unet', 'Imagenes cargadas')

    unet_model = un.build_unet_model(pixels,output_chanels)
    unet_model.summary()

    if callbacks:
        callb = [logs.tensorboard('unet_chan' + str(output_chanels) + '_' + name),  logs.early_stop(10)]
    else:
        callb = []

    if output_chanels in [2,3]:
        log.information('Unet', 'chanels ' + str(output_chanels))
        unet_model.compile(optimizer=tf.keras.optimizers.Adam(),
                        loss="sparse_categorical_crossentropy",
                        metrics=["accuracy", "mean_squared_error"])
        

    if output_chanels == 1:
        log.information('Unet', 'chanels ' + str(output_chanels))
        unet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                        loss=ex.dice_coef_loss,
                        metrics=ex.dice_coef)
    
    
    history = unet_model.fit(images,masks,
                                    batch_size = batch,
                                    epochs = epoch,
                                    callbacks= callb,
                                    shuffle = True,
                                    validation_split = 0.2)

unet_model.save('/home/mr1142/Documents/Data/models/unet_' + str(output_chanels) + '_' + name +'.h5')