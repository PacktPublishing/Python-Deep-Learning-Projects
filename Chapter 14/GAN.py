"""This module contains the DCGAN components."""
from keras.layers import Input, Conv2D, AveragePooling2D
from keras.layers import UpSampling2D, Flatten, Activation, BatchNormalization
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU


def img_generator(input_shape):
    """Generator."""
    generator = Sequential()
    generator.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    generator.add(BatchNormalization())
    generator.add(Activation('relu'))
    generator.add(AveragePooling2D(pool_size=(2, 2)))
    generator.add(Conv2D(64, (3, 3), padding='same'))
    generator.add(BatchNormalization())
    generator.add(Activation('relu'))
    generator.add(AveragePooling2D(pool_size=(2, 2)))
    generator.add(Conv2D(128, (3, 3), padding='same'))
    generator.add(BatchNormalization())
    generator.add(Activation('relu'))
    generator.add(Conv2D(128, (3, 3), padding='same'))
    generator.add(Activation('relu'))
    generator.add(UpSampling2D((2, 2)))
    generator.add(Conv2D(64, (3, 3), padding='same'))
    generator.add(Activation('relu'))
    generator.add(UpSampling2D((2, 2)))
    generator.add(Conv2D(1, (3, 3), activation='tanh', padding='same'))
    return generator


def img_discriminator(input_shape):
    """Discriminator."""
    discriminator = Sequential()
    discriminator.add(Conv2D(64, (3, 3), strides=2, padding='same',
                      input_shape=input_shape))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.2))
    discriminator.add(Conv2D(128, (3, 3), strides=2, padding='same'))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.2))
    discriminator.add(Conv2D(256, (3, 3), padding='same'))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.2))
    discriminator.add(Flatten())
    discriminator.add(Dense(1, activation='sigmoid'))
    return discriminator


def dcgan(discriminator, generator, input_shape):
    """DCGAN."""
    discriminator.trainable = False
    # Accepts the noised input
    gan_input = Input(shape=input_shape)
    # Generates image by passing the above received input to the generator
    gen_img = generator(gan_input)
    # Feeds the generated image to the discriminator
    gan_output = discriminator(gen_img)
    # Compile everything as a model with binary crossentropy loss
    gan = Model(inputs=gan_input, outputs=gan_output)
    return gan
