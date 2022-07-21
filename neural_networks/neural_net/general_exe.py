import argparse
import os
from extra_functions import dice_coef_loss
import logs
import tensorflow as tf
import logging_function as log


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--device',
                        help="GPU device",
                        type=str,
                        default=2)
    parser.add_argument('-c',
                        '--callbacks',
                        type=bool,
                        default=False,
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
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        default="unet",
                        help="type of model") 

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    callbacks = args.callbacks
    name = args.name
    clahe = args.clahe
    model = args.model
    path = '/home/mr1142/Documents/Data/segmentation'
    batch = 8
    epoch = 200
    pixels = 256

    #----------------------------------------------------
    import image_funct as im
    import unet_doble_loss as u_loss
    import unet_effic_funct as u_eff
    import unet_funct as u_net
    import extra_functions as ex

    def unet():
        unet_model = u_net.build_unet_model(256,1)
        unet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                        loss=ex.dice_coef_loss,
                        metrics=ex.dice_coef)
        return unet_model


    def ueff():
        unet_model = u_eff.definitive_model(256, 200)
        unet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                        loss=ex.dice_coef_loss,
                        metrics=ex.dice_coef)
        return unet_model

    def uloss():
        unet_model = u_loss.build_unet_model(256)
        unet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-4),
                            loss=u_loss.MyLoss,
                            metrics = [ex.dice_coef_loss, u_loss.loss_mask])
        return unet_model
    #----------------------------------------------------

    masks_name = os.listdir(os.path.join(path, 'mascara'))

    masks = im.create_tensor(path, 'mascara', masks_name, im.binarize, pixels)
    if clahe:
        images = im.create_tensor(path, 'images', masks_name, im.norm_clahe, pixels)
    else:
        images = im.create_tensor(path, 'images', masks_name, im.normalize, pixels)
    log.information('Unet', 'Imagenes cargadas')

    if callbacks:
        callb = [logs.tensorboard(model + '_' + name), logs.early_stop(10)]
    else:
        callb = []

    images, masks = im.double_tensor(images,masks)

    if model == 'unet':
        unet_model = unet()
    elif model == 'ueff':
        unet_model = ueff()
    elif model == 'uloss':
        unet_model = uloss()
    else:
        unet_model = None
        print('Non correct model')

    history = unet_model.fit(images,masks,
                            batch_size = batch,
                            epochs = epoch,
                            callbacks= callb,
                            shuffle = True,
                            validation_split = 0.2)        

    unet_model.save('/home/mr1142/Documents/Data/models/' + model + '_' + name + '.h5')