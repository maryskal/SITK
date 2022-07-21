import datetime
import os
import tensorflow as tf

def tensorboard(name):
    log_dir = "/home/mr1142/Documents/Data/logs/fit/" + name # datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                        update_freq='batch',
                                                        histogram_freq=1)
    return tensorboard_callback

def weights(name):
    checkpoint_path = "/home/mr1142/Documents/Data/model_weights/" + name + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/{epoch:04d}.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=0)
    return cp_callback
                                        
def early_stop(patient):
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", restore_best_weights=True, patience = patient)
    return early_stop
