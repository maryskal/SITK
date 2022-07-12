from tensorflow.keras import backend as K


def dice_coef(y_true, y_pred, smooth=100):        
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def MyLoss(Y_train, Y_train2, outputs1, outputs2):
    # Loss 1
    loss1 = 1-dice_coef(Y_train, outputs1)
    # Loss 2
    loss2 = abs(Y_train2 - outputs2)
    loss = loss1 + loss2
    return loss1