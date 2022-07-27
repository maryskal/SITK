import numpy as np
import cv2
import imutils
import os
import albumentations as A


def recolor_resize(img, pix=256):
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        print('', end = '')
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
    # return (img - np.mean(img))/ np.std(img)
    return img/np.max(img)

def binarize(img):
    img[img>0] = 1
    return img


def norm_clahe(img):
    img = clahe(img)
    img = normalize(img)
    return img


def apply_to_tensor(tensor, func):
    for i in range(tensor.shape[0]):
        tensor[i,...] = func(tensor[i,...])
    return tensor


def create_tensor(path, folder, names, pixels=256):
    tensor = np.zeros((len(names), pixels,pixels,1))
    for i in range(len(names)):
        tensor[i, ...] = read_img(path, folder, names[i], pixels)
    return tensor
    

def albumentation(input_image, input_mask):
    input_image[input_image<0] = 0
    transform = A.Compose([
        A.RandomGamma (gamma_limit=(20, 200), eps=None, always_apply=False, p=1),
        A.ElasticTransform(alpha=0.5, sigma=20, alpha_affine=20, interpolation=2, border_mode=None, always_apply=False, p=1),
        A.MotionBlur(blur_limit=5, always_apply=False, p=0.2),
        A.Rotate(limit=60, border_mode = None, interpolation=2, p=1),
        A.RandomCrop(p=0.2, width=250, height=250),
        A.Sharpen(alpha=(0, 1), lightness=(0, 1.0), always_apply=False, p=0.1),
    ])
    transformed = transform(image=input_image.astype(np.float32), mask=input_mask.astype(np.float32))
    input_image = recolor_resize(transformed['image'])
    input_mask = recolor_resize(transformed['mask'])
    return input_image, input_mask


def augment(input_image, input_mask):
    r = np.random.randint(-60,60)
    # Random flipping of the image and mask
    input_image = np.expand_dims(imutils.rotate(input_image, angle=r),  axis=-1)
    input_mask = np.expand_dims(imutils.rotate(input_mask, angle=r), axis=-1)
    input_mask = binarize(input_mask)
    return input_image, input_mask


def augment_tensor(images_tensor, masks_tensor, type_augment='', n=2):
    new_n = images_tensor.shape[0]
    pixels = images_tensor.shape[1]
    for _ in range(n):
        new_img = np.zeros((new_n, pixels,pixels,1))
        new_mask = np.zeros((new_n, pixels,pixels,1))
        for j in range(new_n):
            if type_augment == 'old':
                img, mask = augment(images_tensor[j], masks_tensor[j])
            else:
                img, mask = albumentation(images_tensor[j], masks_tensor[j])
            new_img[j, ...] = img
            new_mask[j,...] = mask
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