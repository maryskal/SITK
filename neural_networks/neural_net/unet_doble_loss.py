import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import unet_funct as un
import extra_functions as ex


mask_model = keras.models.load_model('/home/mr1142/Documents/Data/models/mask_1.h5', 
                                    custom_objects={"dice_coef_loss": ex.dice_coef_loss, "dice_coef": ex.dice_coef})
sub_mask = tf.keras.Model(inputs=mask_model.input, outputs=mask_model.layers[18].output)
sub_mask.trainable = False


def loss_mask(y_true, y_pred):
    y_pred = sub_mask(y_pred)
    y_true = sub_mask(y_true)
    return 0.6*abs(y_true - y_pred)


def MyLoss(y_true, y_pred):
    # Loss 1
    loss1 = ex.dice_coef_loss(y_true, y_pred)
    # Loss 2
    loss2 = loss_mask(y_true, y_pred)
    loss = loss1 + loss2
    return loss

def build_unet_model(pixels):
    # inputs
    input = layers.Input(shape=(pixels,pixels,1))
        
    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = un.downsample_block(input, 64)
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
    output = layers.Conv2D(1, 1, padding="same", activation = "sigmoid", name = 'outputs2')(u9)
    
    unet_model = tf.keras.Model(inputs=input, outputs= output , name="U-Net")
        
    return unet_model