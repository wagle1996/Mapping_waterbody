import cv2
import numpy as np
import hyperparameters as hp
import glob
import matplotlib.pyplot as plt
import rasterio as rio
from tensorflow.keras.preprocessing.image import  ImageDataGenerator

def load_data(path, is_mask = False):
    print("Loading data from {}...".format(path))
    filepaths = glob.glob(path + '*.tif')

    images = []
    for path in filepaths:
        with rio.open(path) as f:
            image = f.read()
            image = np.transpose(image, [1,2,0])

            if is_mask:
                image = preprocess_mask(image)
                if 128 in image:
                    image[image == 128] = 0 # set all invalid values to 0
            else:
                image = preprocess_image(image)
            
            images.append(image)
    
    return np.array(images)

def create_generators(images, masks):
    datagen_args = dict(
        rotation_range=0.05,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True
    )

    image_datagen = ImageDataGenerator(**datagen_args)
    mask_datagen = ImageDataGenerator(**datagen_args)

    seed = 0
    image_generator = image_datagen.flow(
        images, 
        batch_size=hp.batch_size,
        seed=seed
    )
    mask_generator = mask_datagen.flow(
        masks,
        batch_size=hp.batch_size,
        seed=seed
    )

    return zip(image_generator, mask_generator)


def preprocess_image(image):
    image = resize_to_fit(image)
    image = image.astype('float64') / 255
    return image

def preprocess_mask(mask):
    mask = resize_to_fit(mask)
    # the input is mxn, so we add another 1d axis to make it mxnx1
    mask = mask.astype('uint8')
    mask = np.expand_dims(mask, axis=-1)

    return mask


def resize_to_fit(img):
    return cv2.resize(img, (hp.img_width, hp.img_width), interpolation=cv2.INTER_LINEAR_EXACT)


if __name__=="__main__":
    image_path = "train/images/"
    mask_path = "train/masks/"

    images = load_data(image_path)
    masks = load_data(mask_path, is_mask = True)

    print(images.shape)
    print(masks.shape)

    plt.imshow(images[0])
    plt.show()
    plt.imshow(masks[0])
    plt.show()

    # image = cv2.imread(image_path + 'Pad10.tif')
    # print(image.shape)
    # plt.imshow(image)
    # plt.show()

    # with rio.open(mask_path + 'Pad10.tif') as f:
    #     mask = f.read()
    #     mask = np.transpose(mask, [1,2,0])
    #     plt.imshow(mask)
    #     plt.show()

    #     print(mask.shape)
    #     mask = preprocess_mask(mask)
    #     print(mask.shape)
    #     plt.imshow(mask[:,:,0])
    #     plt.show()
    #     plt.imshow(mask[:,:,1])
    #     plt.show()