import numpy as np
import cv2
import imutils
import os


def recolor_resize(img, pix):
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        print('error B&W')
    img = cv2.resize(img, (pix, pix))
    img = np.expand_dims(img, axis=-1)
    return img


def read_img(path, folder, img, pix = 256):
    img = cv2.imread(os.path.join(path, folder, img))
    img = recolor_resize(img, pix)
    return img


def clahe(img):
    clahe = cv2.createCLAHE()
    final_img = clahe.apply(img)
    final_img = np.expand_dims(final_img, axis=-1)
    return final_img


def normalize(img):
    return (img - np.mean(img))/ np.std(img)


def binarize(img):
    img[img>0] = 1
    return img


def norm_clahe(img):
    img = clahe(img)
    img = normalize(img)
    return img


def create_tensor(path, folder, names, func, pixels=256):
    tensor = np.zeros((len(names), pixels,pixels,1))
    for i in range(len(names)):
        tensor[i, ...] = func(read_img(path, folder, names[i], pixels))
    return tensor
    

def augment(input_image, input_mask):
    r = np.random.randint(-60,60)
    # Random flipping of the image and mask
    input_image = np.expand_dims(imutils.rotate(input_image, angle=r),  axis=-1)
    input_mask = np.expand_dims(imutils.rotate(input_mask, angle=r), axis=-1)
    input_mask = binarize(input_mask)
    return input_image, input_mask


def double_tensor(images_tensor, masks_tensor):
    pixels = images_tensor.shape[1]
    new_img = np.zeros((images_tensor.shape[0], pixels,pixels,1))
    new_mask = np.zeros((images_tensor.shape[0], pixels,pixels,1))
    for i in range(images_tensor.shape[0]):
        img, mask = augment(images_tensor[i], masks_tensor[i])
        new_img[i, ...] = img
        new_mask[i,...] = mask
    images_tensor = np.concatenate((new_img, images_tensor), axis = 0)
    masks_tensor = np.concatenate((new_mask, masks_tensor), axis = 0)
    return images_tensor, masks_tensor


def apply_mask(img, model):
    pix = img.shape[1]
    img_2 = normalize(recolor_resize(img, 256))
    mask = model.predict(img_2[np.newaxis,...])[0,...]
    mask = cv2.resize(mask, (pix, pix))
    img[mask!=1]=0
    return img