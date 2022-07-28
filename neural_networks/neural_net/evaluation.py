import os
import image_funct as im
import extra_functions as ex
import pandas as pd
import re
import numpy as np

masks_name = ex.list_files(os.path.join('/home/mr1142/Documents/Data/segmentation/splited/validation', 'mascara'))

def evaluate(model, file_names=masks_name):
    path = '/home/mr1142/Documents/Data/segmentation/splited/validation'
    pixels = 256
    masks = im.apply_to_tensor(im.create_tensor(path, 'mascara', file_names, pixels), im.binarize)
    images = im.apply_to_tensor(im.create_tensor(path, 'images', file_names, pixels), im.normalize)
    results = model.evaluate(images, masks, batch_size=8)
    print(results)
    return results


def save_eval(type, name, results, group=''):
    path = '/home/mr1142/Documents/Data/models/validation_results/validation_results' + group + '.csv'
    df = pd.read_csv(path)
    save = [type, name] + results
    df.loc[len(df.index)] = save
    df.to_csv(path, index = False)


def all_evaluations(type, name, model):
    save_eval(type, name, evaluate(model))
    df = pd.read_csv('/home/mr1142/Documents/Data/segmentation/splited/validation/validation_data.csv')
    labels = [re.split('[|]', df['Finding Labels'][i]) for i in df.index]
    labels = [x for xs in labels for x in xs]
    labels = list(np.unique(labels))
    for lab in labels:
        index_contain = [i for i in df.index if bool(re.search(lab, df['Finding Labels'][i]))]
        images = list(df['Image Index'].iloc[index_contain])
        try:
            save_eval(type, name, evaluate(model,images), lab)
        except:
            print(f'model {type} {name}, not saved in lab {lab}')
