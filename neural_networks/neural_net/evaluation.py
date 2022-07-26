import os
import image_funct as im
import extra_functions as ex
import pandas as pd

def evaluate(model):
    path = '/home/mr1142/Documents/Data/segmentation/splited/validation'
    masks_name = ex.list_files(os.path.join(path, 'mascara'))
    pixels = 256
    masks = im.create_tensor(path, 'mascara', masks_name, im.binarize, pixels)
    images = im.create_tensor(path, 'images', masks_name, im.normalize, pixels)
    results = model.evaluate(images, masks, batch_size=8)
    print(results)
    return results

def save_eval(type, name, results):
    path = '/home/mr1142/Documents/Data/models/validation_results.csv'
    df = pd.read_csv(path)
    save = [type, name] + results
    df.loc[len(df.index)] = save
    df.to_csv(path, index = False)

