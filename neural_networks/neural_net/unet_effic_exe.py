from gc import callbacks
import image_funct as im
import unet_effic_funct as un
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
                        default=False,
                        help="Callbacks")
    parser.add_argument('-n',
                        '--name',
                        type=str,
                        default='chanels_model_',
                        help="name of the model") 


    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    path = str(args.path)
    batch = args.batch_size
    epoch = args.epochs
    pixels = args.pixels
    callbacks = args.callbacks
    name = args.name

    masks_name = os.listdir(os.path.join(path, 'mascara'))

    masks = im.create_tensor(path, 'mascara', masks_name, im.binarize, pixels)
    images = im.create_tensor(path, 'images', masks_name, im.normalize, pixels)
    log.information('Unet', 'Imagenes cargadas')

    images, masks = im.double_tensor(images,masks)
    log.information('Unet', 'Nuevas imágenes generadas')

    unet_model = un.definitive_model(pixels, 200)
    unet_model.summary()


    unet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                    loss=ex.dice_coef_loss,
                    metrics=ex.dice_coef)
    log.information('Unet', 'Modelo compilado')


    if callbacks:
        callb = [logs.tensorboard('U_net_efficient_python_'), logs.early_stop(10)]
        history = unet_model.fit(images,masks,
                                batch_size = batch,
                                epochs = epoch,
                                callbacks= callb,
                                shuffle = True,
                                validation_split = 0.2)
    else:
        history = unet_model.fit(images,masks,
                                batch_size = batch,
                                epochs = epoch,
                                shuffle = True,
                                validation_split = 0.2)

unet_model.save('/home/mr1142/Documents/Data/models/' + name + '.h5')