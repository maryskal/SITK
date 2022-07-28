import pandas as pd
import tensorflow as tf
import os
import re
os.environ['CUDA_VISIBLE_DEVICES'] = str(2)

import extra_functions as ex
import unet_doble_loss as u_loss
import unet_funct as u_net
import evaluation as ev


path = '/home/mr1142/Documents/Data/models/validation_results'
csvs = ex.list_files(path)

for csv in csvs:
    df = pd.read_csv(os.path.join(path, csv))
    index = [i for i in df.index if bool(re.search('reevaluation', df['name'][i]))]
    df = df.drop(index)
    df.to_csv(os.path.join(path, csv), index = False)



path = '/home/mr1142/Documents/Data/models'
names = ex.list_files(path)
names = [name for name in names if bool(re.search('validation', name))]
metrics = [ex.dice_coef_loss, u_loss.loss_mask, 'accuracy', 'AUC',
                tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()]

for model in names:
    if bool(re.search('uloss', model)):
        unet_model = tf.keras.models.load_model(os.path.join(path, model), 
                                     custom_objects={"MyLoss": u_loss.MyLoss, 
                                                     "loss_mask": u_loss.loss_mask, 
                                                     "dice_coef_loss": ex.dice_coef_loss,
                                                     "dice_coef": ex.dice_coef})
        unet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-4),
                            loss=u_loss.MyLoss,
                            metrics =metrics)
        ev.all_evaluations('uloss', model[4:-3] + '_reevaluation', unet_model)
    else:
        unet_model = tf.keras.models.load_model(os.path.join(path, model), 
                                     custom_objects={"dice_coef_loss": ex.dice_coef_loss, 
                                                    "dice_coef": ex.dice_coef,
                                                    "loss_mask": u_loss.loss_mask})
        unet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                        loss=ex.dice_coef_loss,
                        metrics=metrics)
        ev.all_evaluations('unet', model[4:-3] + '_reevaluation', unet_model)




import numpy as np
path = '/home/mr1142/Documents/Data/models/validation_results/validation_results' + '.csv'
df = pd.read_csv(path)

evaluations = list(df.columns[2:9])

for ev in evaluations:
    print(ev)
    print('unet')
    np.mean(df[ev][df.type == 'unet'])
    print('uloss') 
    np.mean(df[ev][df.type == 'uloss'])
    print('-----')