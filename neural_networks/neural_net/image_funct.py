import numpy as np
import cv2
import imutils
import os

def read_img(path, folder, img, pixels = 256):
    img = cv2.imread(os.path.join(path, folder, img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (pixels, pixels))
    img = np.expand_dims(img, axis=-1)
    return img


def normalize(img):
    return (img - np.mean(img))/ np.std(img)


def binarize(img):
    img[img>0] = 1
    return img

def clahe(img):
    clahe = cv2.createCLAHE()
    final_img = clahe.apply(img)
    return img


def create_tensor(path, folder, names, func, pixels=256):
    tensor = np.zeros((len(names), pixels,pixels,1))
    for i in range(len(names)):
        tensor[i, ...] = func(clahe(read_img(path, folder, names[i], pixels)))
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