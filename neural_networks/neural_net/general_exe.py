import argparse
import os
import logs
import tensorflow as tf
import evaluation as ev


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
    parser.add_argument('-a',
                        '--augmentation',
                        type=int,
                        default=3,
                        help="number of replications")
    parser.add_argument('-ty',
                        '--type_aug',
                        type=str,
                        default='old',
                        help="type_augmentation")                    

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    callbacks = args.callbacks
    name = args.name
    clahe = args.clahe
    model = args.model
    augmentation = args.augmentation
    type_augmentation = args.type_aug
    path = '/home/mr1142/Documents/Data/segmentation/splited/train'
    batch = 8
    epoch = 200
    pixels = 256

    #----------------------------------------------------
    import image_funct as im
    import unet_doble_loss as u_loss
    import unet_effic_funct as u_eff
    import unet_funct as u_net
    import extra_functions as ex

    metrics = [ex.dice_coef_loss, u_loss.loss_mask, 'accuracy', 'AUC',
                tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()]

    def unet():
        unet_model = u_net.build_unet_model(256,1)
        unet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                        loss=ex.dice_coef_loss,
                        metrics=metrics)
        return unet_model

    def ueff():
        unet_model = u_eff.definitive_model(256, 200)
        unet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                        loss=ex.dice_coef_loss,
                        metrics=metrics)
        return unet_model

    def uloss():
        unet_model = u_loss.build_unet_model(256)
        unet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-4),
                            loss=u_loss.MyLoss,
                            metrics =metrics)
        return unet_model
    #----------------------------------------------------

    # DATOS
    masks_name = ex.list_files(os.path.join(path, 'mascara'))
    images = im.create_tensor(path, 'images', masks_name, pixels)
    masks = im.create_tensor(path, 'mascara', masks_name, pixels)

    # Aumento
    images, masks = im.augment_tensor(images,masks,type_augmentation,augmentation)

    # Binarize and normalize
    if clahe:
        images = im.apply_to_tensor(images, im.norm_clahe) 
    else:
        images = im.apply_to_tensor(images, im.normalize) 
    masks = im.apply_to_tensor(masks, im.binarize) 

    # CALLBACKS
    if callbacks:
        callb = [logs.tensorboard(model + '_' + name), logs.early_stop(10)]
    else:
        callb = []

    # MODELOS
    if model == 'unet':
        unet_model = unet()
    elif model == 'ueff':
        unet_model = ueff()
    elif model == 'uloss':
        unet_model = uloss()
    else:
        unet_model = None
        print('\n INCORRECT MODEL \n')

    # ENTRENAMIENTO
    history = unet_model.fit(images,masks,
                            batch_size = batch,
                            epochs = epoch,
                            callbacks= callb,
                            shuffle = True,
                            validation_split = 0.2)        

    unet_model.save('/home/mr1142/Documents/Data/models/' + model + '_' + name + '.h5')
    
    # EVALUACIÓN
    ev.all_evaluations(model, name, unet_model)