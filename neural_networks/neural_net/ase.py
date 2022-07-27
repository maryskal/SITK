import extra_functions as ex
import re
import tensorflow as tf
import unet_doble_loss as u_loss
import unet_funct as u_net
import evaluation as ev


import pandas as pd
import os

os.environ['CUDA_VISIBLE_DEVICES'] = str(3)


path = '/home/mr1142/Documents/Data/models/validation_results'
csvs = ex.list_files(path)

for csv in csvs:
    df = pd.read_csv(os.path.join(path, csv))
    index = [i for i in df.index if bool(re.search('old', df['name'][i]))]
    df = df.drop(index)
    df.to_csv(os.path.join(path, csv), index = False)



path = '/home/mr1142/Documents/Data/models'
names = ex.list_files(path)
names = [name for name in names if bool(re.search('old', name))]
metrics = [ex.dice_coef_loss, u_loss.loss_mask, 'accuracy', 'AUC',
                tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()]

uloss = 0
unet = 0
for model in names:
    if bool(re.search('uloss', model)):
        uloss += 1
        unet_model = u_loss.build_unet_model(256)
        unet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-4),
                            loss=u_loss.MyLoss,
                            metrics =metrics)
        ev.all_evaluations('uloss', 'old_aumento-validation_' +  str(uloss), unet_model)
    else:
        unet += 1
        unet_model = u_net.build_unet_model(256,1)
        unet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                        loss=ex.dice_coef_loss,
                        metrics=metrics)
        ev.all_evaluations('unet', 'old_aumento-validation_'+ str(unet), unet_model)




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