import os

from PIL import Image
import tensorflow as tf
from tensorflow.keras import Input
import numpy as np
import matplotlib.pyplot as plt
import argparse

from model import UNet
from preprocessing import create_generators, load_data
import hyperparameters as hp

image_path = "Datazoo/TrainImages/"
mask_path = "Datazoo/TrainLabels/"
test_path = "Datazoo/TestImages/"
test_mask = "Datazoo/TestLabels/"

checkpoint_path = "checkpoints/"

parser = argparse.ArgumentParser(
    description="Watermap!",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--skip_train',
    action='store_true',
    help="If true, skips training.")
parser.add_argument(
    '--augment_data',
    action='store_true',
    help="If true, uses data augmentation during training.")
parser.add_argument(
    '--load_checkpoint',
    default=None,
    help="Path to model checkpoint (.hdf5 file)")
parser.add_argument(
    '--show_example',
    action='store_true',
    help="If true, shows example output in comparison to expected output.")
parser.add_argument(
    '--save_results',
    action='store_true',
    help="If true, saves trained model outputs for images in training/test set.")
ARGS = parser.parse_args()

if __name__ == "__main__":
    # create model
    unet = UNet()
    inputs = Input(shape=(hp.img_width, hp.img_height, hp.img_channels))
    outputs = unet.call(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.summary()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"])
    
    # load data
    # images should be size [m, img_width, img_height, img_channels]
    # masks should be size [m, img_width, img_height, mask_channels]
    train_images = load_data(image_path)
    train_masks = load_data(mask_path, is_mask=True)
    test_images = load_data(test_path)
    test_masks = load_data(test_mask, is_mask=True)
    train_generator = create_generators(train_images, train_masks)
    
    # create checkpoint callback
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path + "/{epoch:02d}-{accuracy:.3f}.hdf5",
        monitor="val_accuracy",
        save_weights_only=True,
        save_freq="epoch",
        save_best_only=True
    )
    
    # load checkpoint
    if ARGS.load_checkpoint:
        model.load_weights(checkpoint_path + ARGS.load_checkpoint)
    
    # train model
    if not ARGS.skip_train:
        if ARGS.augment_data:
            results = model.fit(train_generator, steps_per_epoch=hp.steps_per_epoch, epochs=hp.epochs,
                validation_data=(test_images, test_masks), callbacks=[checkpoint_callback])    
        else:
            results = model.fit(train_images, train_masks, batch_size=hp.batch_size, epochs=hp.epochs, 
                validation_data=(test_images, test_masks), callbacks=[checkpoint_callback])
        
        # evaluate results
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(results.history["loss"], color='r', label='Training Data')
        ax[0].plot(results.history["val_loss"], color='b', linestyle='dashed', label='Validation Data')
        ax[0].set_title('Loss Over Time')
        ax[0].legend()
        ax[1].plot(results.history["accuracy"], color='r', label='Training Data')
        ax[1].plot(results.history["val_accuracy"], color='b', linestyle='dashed', label='Validation Data')
        ax[1].set_title('Accuracy Over Time')
        ax[1].legend()
        plt.show()

    # evaluate model
    model.evaluate(train_images, train_masks)
    model.evaluate(test_images, test_masks)
    
    if ARGS.show_example:    
        # visualize specific examples
        index = 7
        image = train_images[index]
        mask = train_masks[index]
        pred_mask = model.predict(image[np.newaxis, ...])
        pred_mask = tf.argmax(pred_mask[0], axis=-1)[..., tf.newaxis]
        
        test_image = test_images[index]
        test_mask = test_masks[index]
        test_pred_mask = model.predict(test_image[np.newaxis, ...])
        test_pred_mask = tf.argmax(test_pred_mask[0], axis=-1)[..., tf.newaxis]
        
        fig, ax = plt.subplots(2, 3)
        ax[0, 0].imshow(image)
        ax[0, 0].set_title("Processed Image")
        ax[0, 1].imshow(mask)
        ax[0, 1].set_title("Real Water Mask")
        ax[0, 2].imshow(pred_mask)
        ax[0, 2].set_title("Predicted Water Mask")
        
        ax[1, 0].imshow(test_image)
        ax[1, 0].set_title("Test Processed Image")
        ax[1, 1].imshow(test_mask)
        ax[1, 1].set_title("Test Real Water Mask")
        ax[1, 2].imshow(test_pred_mask)
        ax[1, 2].set_title("Test Predicted Water Mask")
        
        plt.show()

    if ARGS.save_results:        
        # run the model on train label and test label, save the results in tif format and same name as the original image
        # and store the results in TrainPredMask and TestPredMask directory respectively
        # TrainPredMask Path: Datazoo/TrainPredMask
        # TestPredMask Path: Datazoo/TestPredMask
        trainPredMaskPath = "Datazoo/TrainPredMask/"
        testPredMaskPath = "Datazoo/TestPredMask/"
        
        # list all train label files' name in mask_path without extension
        train_label_names = [f[:-4] for f in os.listdir(mask_path) if f.endswith(".tif")]
        print(train_label_names[:5])
        test_label_names = [f[:-4] for f in os.listdir(test_mask) if f.endswith(".tif")]
        print(test_label_names[:5])
        
        
        def run_model_and_store_predicted_masks(image_arr, name_arr, dir_path):
            for image, name in zip(image_arr, name_arr):
                pred_mask = model.predict(image[np.newaxis, ...])
                pred_mask = tf.argmax(pred_mask[0], axis=-1)[..., tf.newaxis]
                # convert to tif image
                pred_mask = pred_mask.numpy()
                pred_mask = pred_mask * 255
                pred_mask = np.squeeze(pred_mask)
                pred_mask = pred_mask.astype(np.uint8)
                pred_mask = Image.fromarray(pred_mask)
                pred_mask.save(dir_path + name + ".tif")  # store the predicted mask in tif format
        
        
        run_model_and_store_predicted_masks(train_images, train_label_names, trainPredMaskPath)
        run_model_and_store_predicted_masks(test_images, test_label_names, testPredMaskPath)
    

