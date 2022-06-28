from tensorflow import keras
from tensorflow import nn

class CustomCallback(keras.callbacks.Callback):
    
    def __init__(self,x_test,y_test,writer,max_img,batch_step=50,delta=0,patience=3):
        self.steps = 0
        self.x_test = x_test
        self.y_test = y_test
        self.writer = writer
        self.max_img = max_img
        self.batch_loss_queue = []
        
        self.batch_step = batch_step
        self.delta = delta
        self.patience = patience

    def on_epoch_end(self, epoch, logs=None):
        

        prediction = self.model.predict(self.x_test)
        prediction_test = np.stack((self.x_test,
                                    prediction,self.y_test),axis=(-3)).reshape((-1,self.x_test.shape[-3],self.x_test.shape[-3]*3,1))
        with self.writer.as_default():
            tf.summary.image('Image',data = prediction_test, step=epoch, max_outputs=self.max_img)


    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        values = list(logs.values())

        if batch % self.batch_step == 0:
            
            self.steps += 1
            with self.writer.as_default():
                tf.summary.scalar('batch_loss',data = values[0], step=self.steps*self.batch_step)
                
        if self.steps >= self.patience:
            self.model.stop_training = self.early_stopping(values[0])
        else:
            self.batch_loss_queue.append(values[0])
                
    def early_stopping(self,batch_loss):
        
        self.batch_loss_queue.pop(0)
        self.batch_loss_queue.append(batch_loss)
        diff = np.zeros((len(self.batch_loss_queue) - 1)).astype(bool)
        for i in range(len(self.batch_loss_queue) - 1):
            diff[i] = (np.absolute(self.batch_loss_queue[i+1] - self.batch_loss_queue[i]) < self.delta)


        return all(diff)
        
def custom_loss(y_actual,y_pred):
    # Mean Absolute Error
    mae = tf.keras.losses.MeanAbsoluteError()
    
    # Noise variance
    noise = y_actual - y_pred
    noise_mean_2 = nn.conv2d(noise,np.ones((3,3,1,1))/9,strides=[1,1],padding='SAME')**2
    noise_2_mean = nn.conv2d(noise**2,np.ones((3,3,1,1))/9,strides=[1,1],padding='SAME')
    noise_var = tf.reduce_mean((noise_2_mean - noise_mean_2)[:,1:-1,1:-1,:])
    custom_loss = mae(y_actual,y_pred) + noise_var
    
    return custom_loss            
    

if __name__ == '__main__':
    import os
    from pathlib import Path
    import numpy as np
    import argparse
    import SimpleITK as sitk
    import tensorflow as tf
    from tensorflow.keras import optimizers, losses
    from tensorflow.keras.utils import plot_model
    from utils import print_args
    from networks import unet_harmonization
    from DataGenerator import DataGenerator

    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--device',
                        help="GPU device",
                        type=int,
                        default=2)
    parser.add_argument('-v',
                        '--verbose',
                        help="Verbose level",
                        type=int,
                        default=2)
    parser.add_argument('-e',
                        '--epochs',
                        type=int,
                        default=100,
                        help="Number of Epochs")
    parser.add_argument('-cbn',
                        '--callback_num',
                        type=int,
                        default=4,
                        help="Number of images to show in tensorboard")
    parser.add_argument('-bs',
                        '--batch_size',
                        type=int,
                        default=200,
                        help="Batch size")
    parser.add_argument('-act',
                        '--activation',
                        type=str,
                        default='swish',
                        help="Activation (default swish)")
    parser.add_argument('-lr', '--lr', type=float, default=5e-5)
    parser.add_argument('-da',
                        '--daugmention',
                        action='store_true',
                        help="Data Augmentation")
    parser.add_argument('-path_X',
                        '--path_X',
                        type=str,
                        help="X data for model")
    parser.add_argument('-path_Y',
                        '--path_Y',
                        type=str,
                        help="Y data for model")
    

    # Read data and exclude Undefine patches
    args = parser.parse_args()
    assert ((args.device < 4) and (args.device >= 0))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    # Kera

    args.gpus = tf.config.list_physical_devices('GPU')
    DG = DataGenerator(args.path_X,args.path_Y,args.callback_num,args.batch_size)

    # Create and train the model
    m_name = 'Jagoba_Deep_Harmonization'

    m_name += '_d%i_e%i_%s' % (args.device, args.epochs, args.activation)
    m_name += '_lr{:.0E}'.format(args.lr)
    
    date_time = str(np.datetime64('now'))
    logs_name = "".join(list(map(lambda c: c if (c >= '0' and c <= '9') else '_', date_time)))[5:-3]
    m_name += '_' + logs_name
    
    # Create tensorflow callback
#     log_dir = "logs/fit/" + m_name
#     UpdateFreq = np.floor(DG.num_step_epochs() / 150).astype(int)
#     tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=500)
    
    input_shape = DG.X.shape[-2:] + (1,)
    model = unet_harmonization(input_shape, activation=args.activation)

    save_folder = os.path.join('models', m_name)
    print('Saving files in: {}'.format(save_folder), flush=True)
    save_path = Path(save_folder)
    save_path.mkdir(parents=True, exist_ok=True)

    plot_model(model,
               to_file=save_folder + '/' + m_name + '.png',
               show_shapes=True,
               expand_nested=True)

    
    ## Create tensorflow summary
    tf_summary_writer = tf.summary.create_file_writer("logs/fit/" + m_name + '/metrics')
    MyCallback = CustomCallback(DG.x_callback,DG.y_callback,tf_summary_writer,args.callback_num,batch_step=50,delta=1e-4,patience=9)
    
    ## Optimizer and loss
    optim = optimizers.RMSprop(learning_rate=args.lr)
    loss = losses.MeanAbsoluteError()
    model.compile(optimizer=optim, loss=custom_loss)
    # Train the model
    hist = model.fit(DG.gen_train_batch(),
                     steps_per_epoch=DG.num_step_epochs('train')//3,
                     epochs=args.epochs,
                     validation_data=DG.gen_val_batch(),
                     validation_steps=DG.num_step_epochs('val')//2,
                     callbacks=[MyCallback],
                     verbose=args.verbose)

    # Save the trained model
    np.save(save_folder + '/' + m_name + '_hist.npy', hist.history)
    model.save(save_folder + '/' + m_name, save_format='tf')
