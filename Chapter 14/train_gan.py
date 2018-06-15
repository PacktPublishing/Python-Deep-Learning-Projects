import numpy as np
from training_plots import upscale, generated_images_plot, plot_training_loss
from training_plots import plot_generated_images_combined
from keras.optimizers import Adam
from keras import backend as k
import matplotlib.pyplot as plt
from tqdm import tqdm

from GAN import img_generator, img_discriminator, dcgan

from keras.datasets import mnist
from train_mnist import train_mnist

%matplotlib inline
# Smoothing value
smooth_real = 0.9
# Number of epochs
epochs = 5
# Batchsize
batch_size = 128
# Optimizer for the generator
optimizer_g = Adam(lr=0.0002, beta_1=0.5)
# Optimizer for the discriminator
optimizer_d = Adam(lr=0.0004, beta_1=0.5)
# Shape of the input image
input_shape = (28, 28, 1)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
X_train = (X_train - 127.5) / 127.5
X_test = (X_test - 127.5) / 127.5


def noising(image):
    """Masking."""
    import random
    array = np.array(image)
    i = random.choice(range(8, 12))  # x coord for top left corner of the mask
    j = random.choice(range(8, 12))  # y coord for top left corner of the mask
    array[i:i+8, j:j+8] = -1  # setting the pixels in the masked region to 0
    return array


noised_train_data = np.array([*map(noising, X_train)])
noised_test_data = np.array([*map(noising, X_test)])
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],
                          X_train.shape[2], 1)
noised_train_data = noised_train_data.reshape(noised_train_data.shape[0],
                                              noised_train_data.shape[1],
                                              noised_train_data.shape[2], 1)
noised_test_data = noised_test_data.reshape(noised_test_data.shape[0],
                                            noised_test_data.shape[1],
                                            noised_test_data.shape[2], 1)


def train(X_train,      noised_train_data,
          input_shape,  smooth_real,
          epochs,       batch_size,
          optimizer_g, optimizer_d):
    """Training GAN."""
    discriminator_losses = []
    generator_losses = []

    # Number of iteration possible with batches of size 128
    iterations = X_train.shape[0] // batch_size
    # Load the generator and the discriminator
    generator = img_generator(input_shape)
    discriminator = img_discriminator(input_shape)
    # Compile the discriminator with binary_crossentropy loss
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer_d)
    # Feed the generator and the discriminator to the function dcgan
    # to form the DCGAN architecture
    gan = dcgan(discriminator, generator, input_shape)
    # Compile the DCGAN with binary_crossentropy loss
    gan.compile(loss='binary_crossentropy', optimizer=optimizer_g)

    for i in range(epochs):
        print('Epoch %d' % (i+1))
        # Use tqdm to get an estimate of time remaining
        for j in tqdm(range(1, iterations+1)):
            # batch of original images (batch = batchsize)
            original = X_train[np.random.randint(0, X_train.shape[0],
                                                 size=batch_size)]
            # batch of noised images (batch = batchsize)
            noise = noised_train_data[np.random.randint(0,
                                                        noised_train_data.shape[0],
                                                        size=batch_size)]
            # Generate fake images
            generated_images = generator.predict(noise)
            # Labels for generated data
            dis_lab = np.zeros(2*batch_size)
            dis_train = np.concatenate([original, generated_images])
            # label smoothing for original images
            dis_lab[:batch_size] = smooth_real
            # Train discriminator on original iamges
            discriminator.trainable = True
            discriminator_loss = discriminator.train_on_batch(dis_train,
                                                              dis_lab)
            # save the losses
            discriminator_losses.append(discriminator_loss)
            # Train generator
            gen_lab = np.ones(batch_size)
            discriminator.trainable = False
            sample_indices = np.random.randint(0, X_train.shape[0],
                                               size=batch_size)
            original = X_train[sample_indices]
            noise = noised_train_data[sample_indices]

            generator_loss = gan.train_on_batch(noise, gen_lab)
            # save the losses
            generator_losses.append(generator_loss)
            if i == 0 and j == 1:
                print('Iteration - %d', j)
                generated_images_plot(original, noise, generator)
                plot_generated_images_combined(original, noise, generator)

        print("Discriminator Loss: ", discriminator_loss,
              ", Adversarial Loss: ", generator_loss)
        generated_images_plot(original, noise, generator)
        plot_generated_images_combined(original, noise, generator)
    # plot the losses
    plot_training_loss(discriminator_losses, generator_losses)

    return generator


generator = train(X_train, noised_train_data,
                  input_shape, smooth_real,
                  epochs, batch_size,
                  optimizer_g, optimizer_d)


mnist_model = train_mnist(input_shape, X_train, y_train)

gen_imgs_test = generator.predict(noised_test_data)
gen_pred_lab = mnist_model.predict_classes(gen_imgs_test)

# plot of 10 generated images and their predicted label
fig = plt.figure(figsize=(8, 4))
plt.title('Generated Images')
plt.axis('off')
columns = 5
rows = 2
for i in range(0, rows*columns):
    fig.add_subplot(rows, columns, i+1)
    plt.title('Act: %d, Pred: %d' % (gen_pred_lab[i], y_test[i]))  # label
    plt.axis('off')  # turn off axis
    plt.imshow(upscale(np.squeeze(gen_imgs_test[i])), cmap='gray')  # grayscale
plt.show()

# prediction on the masked images
labels = mnist_model.predict_classes(noised_test_data)
print('The model model accuracy on the masked images is:',
      np.mean(labels == y_test)*100)

# prediction on the generated images
labels = mnist_model.predict_classes(gen_imgs_test)
print('The model model accuracy on the generated images is:',
      np.mean(labels == y_test)*100)
