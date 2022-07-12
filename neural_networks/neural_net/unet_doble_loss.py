import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import unet_funct as un
import extra_functions as ex


def charge_mask():
    mask_model = keras.models.load_model('/home/mr1142/Documents/Data/models/mask_1.h5', 
                                     custom_objects={"dice_coef_loss": ex.dice_coef_loss, "dice_coef": ex.dice_coef})
    sub_mask = tf.keras.Model(inputs=mask_model.input, outputs=mask_model.layers[18].output)
    return sub_mask


def build_unet_model(pixels, mask):
    # inputs
    X_train = layers.Input(shape=(pixels,pixels,1))
    Y_train = layers.Input(shape=(pixels,pixels,1))
    Y_train2 = layers.Input(shape=(16, 16, 1024))
        
    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = un.downsample_block(X_train, 64)
    # 2 - downsample
    f2, p2 = un.downsample_block(p1, 128)
    # 3 - downsample
    f3, p3 = un.downsample_block(p2, 256)
    # 4 - downsample
    f4, p4 = un.downsample_block(p3, 512)

    # 5 - bottleneck
    bottleneck = un.double_conv_block(p4, 1024)

    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = un.upsample_block(bottleneck, f4, 512)
    # 7 - upsample
    u7 = un.upsample_block(u6, f3, 256)
    # 8 - upsample
    u8 = un.upsample_block(u7, f2, 128)
    # 9 - upsample
    u9 = un.upsample_block(u8, f1, 64)

    # outputs
    outputs1 = layers.Conv2D(1, 1, padding="same", activation = "sigmoid", name = 'outputs2')(u9)
    
    # Mask 

    outputs2 = mask(outputs1)
    mask.trainable = False
    
    unet_model = tf.keras.Model(inputs=[X_train, Y_train, Y_train2], outputs=[outputs2, outputs1], name="U-Net")
    
    unet_model.add_loss(ex.MyLoss(Y_train, Y_train2, outputs1, outputs2))
    
    return unet_model