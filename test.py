import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import os

import rasterio as rio

with rio.open('pad.tif') as f:
    image = f.read()
    image = np.transpose(image, [1,2,0])

    print(np.max(image))
    print(np.min(image))
    print(image.shape)
    original_shape = image.shape
    #plt.imshow(image, cmap='gray')
    #plt.show()

# pad_wa_mask.tif
with rio.open('pad_water_m2.tif') as f:
    mask = f.read()
    mask = np.transpose(mask, [1,2,0])

    print(np.max(mask))
    print(np.min(mask))

    new_height = int(original_shape[1] * (300 / 275))
    new_width = int(original_shape[0] * (300/275))
    #mask = cv2.resize(mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR_EXACT)    


    print(mask.shape)
    #plt.imshow(mask, cmap='gray', vmin=0, vmax=1)
    #plt.show()

# image = plt.imread('train/masks/pad6.tif')
# image = image[:,:,0]
# plt.imshow(image, cmap='gray')
# plt.show()


fig, ax = plt.subplots(nrows=1, ncols=2)

ax[0].imshow(image)
ax[1].imshow(mask, vmin=0, vmax=1)
plt.show()


height=256
width=256
image_path = "train\\pad_images"
mask_path = "train\\pad_masks"
k = 0
for i in range(0,original_shape[0],height):
    for j in range(0,original_shape[1],width):

        box = (j, i, j+width, i+height)
        cropped_image = image[j:j+width, i:i+width]
        cropped_mask = mask[j:j+width, i:i+width]

        if -128 not in cropped_mask:
            print("saving image {}".format(k))
            fig, ax = plt.subplots(nrows=1, ncols=2)

            ax[0].imshow(cropped_image)
            ax[1].imshow(cropped_mask)
            plt.show()
            # image_png = Image.fromarray(cropped_image)
            # mask_png = Image.fromarray(cropped_mask)
            # image_png.save(os.path.join(image_path,"IMG-%s.png" % k))
            # mask_png.save(os.path.join(mask_path,"IMG-%s.png" % k))
            cv2.imwrite(os.path.join(image_path,"IMG-%s.png" % k), cropped_image)
            cv2.imwrite(os.path.join(mask_path,"IMG-%s.png" % k), cropped_mask)
            #np.save(os.path.join(image_path,"IMG-%s.png" % k), cropped_image)
            #np.save(os.path.join(mask_path,"IMG-%s.png" % k), cropped_mask)
        k +=1


image = plt.imread('train/pad_images/IMG-85.png')
plt.imshow(image)
plt.show()

mask = plt.imread('train/pad_masks/IMG-85.png')
mask *= 255
plt.imshow(mask, vmin=0, vmax=1)
plt.show()