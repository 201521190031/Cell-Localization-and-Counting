import os
import numpy as np

def label2flux(label):
    """
    Create direction_field of corresponding label

    Returns:
        direction_field: shape (2, h, w), weight_matrix: (h, w)
    """
    label = label.copy()
    label = label.astype(np.int32)
    height, width = label.shape
    label += 1
    categories = np.unique(label)

    if 0 in categories:
        raise ValueError('invalid category')

    label = cv2.copyMakeBorder(label, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    weight_matrix = np.zeros((height+2, width+2), dtype=np.float32)
    direction_field = np.zeros((2, height+2, width+2), dtype=np.float32)

    for category in categories:
        img = (label == category).astype(np.uint8)
        weight_matrix[img > 0] = 1. / np.sqrt(img.sum())

        _, labels = cv2.distanceTransformWithLabels(img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL)

        index = np.copy(labels)
        index[img > 0] = 0
        place =  np.argwhere(index > 0)

        nearCord = place[labels-1,:]
        x = nearCord[:, :, 0]
        y = nearCord[:, :, 1]
        nearPixel = np.zeros((2, height+2, width+2))
        nearPixel[0,:,:] = x
        nearPixel[1,:,:] = y
        grid = np.indices(img.shape)
        grid = grid.astype(float)
        diff = grid - nearPixel    

        direction_field[:, img > 0] = diff[:, img > 0]     

    weight_matrix = weight_matrix[1:-1, 1:-1]
    direction_field = direction_field[:, 1:-1, 1:-1]

    return direction_field, weight_matrix
