import tensorflow as tf
from tensorflow.keras import Model, Sequential, layers
from hyperparameters import mask_channels

class Encoder():
    
    def __init__(self):
        super(Encoder, self).__init__()

        self.skip_layers = []
        
        # encoder architecture
        self.block1 = self.convolution_block(
            n_filters=32,
            filter_size=3,
            dropout_prob=0)
        self.maxpool1 = layers.MaxPooling2D(pool_size=(2,2))
        self.block2 = self.convolution_block(
            n_filters=64,
            filter_size=3,
            dropout_prob=0)
        self.maxpool2 = layers.MaxPooling2D(pool_size=(2,2))
        self.block3 = self.convolution_block(
            n_filters=128,
            filter_size=3,
            dropout_prob=0)
        self.maxpool3 = layers.MaxPooling2D(pool_size=(2,2))
        self.block4 = self.convolution_block(
            n_filters=256,
            filter_size=3,
            dropout_prob=0.3)
        self.maxpool4 = layers.MaxPooling2D(pool_size=(2,2))
        self.block5 = self.convolution_block(
            n_filters=512,
            filter_size=3,
            dropout_prob=0.3)
    
    def call(self, x):
        x = self.block1(x)
        self.skip_layers.append(x)
        x = self.maxpool1(x)
        x = self.block2(x)
        self.skip_layers.append(x)
        x = self.maxpool2(x)
        x = self.block3(x)
        self.skip_layers.append(x)
        x = self.maxpool3(x)
        x = self.block4(x)
        self.skip_layers.append(x)
        x = self.maxpool3(x)
        x = self.block5(x)
        return x
    
    def convolution_block(self, n_filters, filter_size, dropout_prob):
        block_layers = Sequential([])

        # conv 1
        block_layers.add(layers.Conv2D(
            filters=n_filters,
            kernel_size=filter_size,
            strides=1,
            activation="relu",
            padding="same",
            kernel_initializer='HeNormal'
        ))
        
        # conv 2
        block_layers.add(layers.Conv2D(
            filters=n_filters,
            kernel_size=filter_size,
            strides=1,
            activation="relu",
            padding="same",
            kernel_initializer='HeNormal'
        ))
        
        # batch normalization
        block_layers.add(layers.BatchNormalization())

        # dropout
        block_layers.add(layers.Dropout(dropout_prob))
        
        return block_layers


class Decoder():
    
    def __init__(self, num_classes):
        super(Decoder, self).__init__()

        self.num_classes = num_classes

        # decoder architecture
        self.deconv1 = layers.Conv2DTranspose(
            filters=256,
            kernel_size=3,
            strides=(2,2),
            padding="same"
        )
        self.block1 = self.convolution_block(256, 3)
        self.deconv2 = layers.Conv2DTranspose(
            filters=128,
            kernel_size=3,
            strides=(2,2),
            padding="same"
        )
        self.block2 = self.convolution_block(128, 3)
        self.deconv3 = layers.Conv2DTranspose(
            filters=64,
            kernel_size=3,
            strides=(2,2),
            padding="same"
        )
        self.block3 = self.convolution_block(64, 3)
        self.deconv4 = layers.Conv2DTranspose(
            filters=32,
            kernel_size=3,
            strides=(2,2),
            padding="same"
        )
        self.block4 = self.convolution_block(32, 3)  

        self.finalConv = layers.Conv2D(
            filters=32,
            kernel_size=3,
            activation="relu",
            padding="same",
            kernel_initializer='HeNormal'
        )
        self.categorize = layers.Conv2D(
            filters=self.num_classes,
            kernel_size=1,
            padding="same"
        )
    
    def call(self, x, skip_outputs):
        x = self.deconv1(x)
        x = layers.concatenate([x, skip_outputs[-1]], axis=3)
        x = self.block1(x)
        x = self.deconv2(x)
        x = layers.concatenate([x, skip_outputs[-2]], axis=3)
        x = self.block2(x)
        x = self.deconv3(x)
        x = layers.concatenate([x, skip_outputs[-3]], axis=3)
        x = self.block3(x)
        x = self.deconv4(x)
        x = layers.concatenate([x, skip_outputs[-4]], axis=3)
        x = self.block4(x)
        x = self.finalConv(x)
        x = self.categorize(x)
        return x
    
    def convolution_block(self, n_filters, filter_size):
        block_layers = Sequential([])

        # conv 1
        block_layers.add(layers.Conv2D(
            filters=n_filters,
            kernel_size=filter_size,
            strides=1,
            activation="relu",
            padding="same",
            kernel_initializer='HeNormal'
        ))
        
        # conv 2
        block_layers.add(layers.Conv2D(
            filters=n_filters,
            kernel_size=filter_size,
            strides=1,
            activation="relu",
            padding="same",
            kernel_initializer='HeNormal'
        ))

        return block_layers

class UNet(Model):

    def __init__(self):
        super(UNet, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder(mask_channels)

    def call(self, x):
        # encoder
        x = self.encoder.call(x)

        #decoder
        x = self.decoder.call(x, self.encoder.skip_layers)
        
        return x

if __name__=="__main__":
    unet = UNet()
    output = unet(tf.keras.Input(shape=(128, 128, 3)))

    print(output.shape)