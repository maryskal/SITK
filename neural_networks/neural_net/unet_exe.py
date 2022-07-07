from gc import callbacks
import image_funct as im
import unet_funct as un
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


    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    path = str(args.path)
    batch = args.batch_size
    epoch = args.epochs
    pixels = args.pixels

    masks_name = os.listdir(os.path.join(path, 'mascara'))

    masks = im.create_tensor(path, 'mascara', masks_name, pixels, im.normalize)
    images = im.create_tensor(path, 'images', masks_name, pixels, im.binarize)

    images, masks = im.double_tensor(images,masks)
    print(images.shape)
    print(masks.shape)

    unet_model = un.build_unet_model(pixels)
    unet_model.summary()

    callb = [logs.tensorboard('U_net_'), logs.weights('U_net'), logs.early_stop(7)]

    unet_model.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy", "mean_squared_error"])


    history = unet_model.fit(images,masks,
                            batch_size = batch,
                            epochs = epoch,
                            callbacks= callb,
                            shuffle = True,
                            validation_split = 0.2)
