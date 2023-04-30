import tensorflow as tf
import keras
print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)


import numpy as np
from keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Dropout, UpSampling2D, Flatten, Dense, Input
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras import layers
from keras.models import load_model
from keras.layers import InputLayer
import matplotlib.pyplot as plt
from RL import *

class BellaModel:
    def __init__(self):
        self.save_folder = '/work/baskargroup/bella/data_efficient/coms590_rl/report/test_rl/'
        self.path = '/work/baskargroup/bella/data_efficient/data/'
    
    def load_data(self):
        npy_data = np.load(self.path + 'augmented_JF_filtered_norm_train.npy', allow_pickle=True)
        img = npy_data[:,0]
        current = npy_data[:,1]
        ff = npy_data[:,2]
        # reshape data
        ff = ff.reshape((len(current),1))
        current = current.reshape((len(current),1))
        images = []
        for image in img:
            images.append(image)
        images=np.asarray(images)
        data = images.reshape(len(current),128,128,1)
        return data, current, ff

    def save_train_test_data(self, train_image, test_image, train_current, test_current, train_ff, test_ff):
        np.save(self.save_folder + 'train_image', np.asarray(train_image))
        np.save(self.save_folder + 'test_image', np.asarray(test_image))
        np.save(self.save_folder + 'train_current', np.asarray(train_current))
        np.save(self.save_folder + 'test_current', np.asarray(test_current))
        np.save(self.save_folder + 'train_ff', np.asarray(train_ff))
        np.save(self.save_folder + 'test_ff', np.asarray(test_ff))

    def conv2d_block(self, input_tensor, filters, kernel_size, batch_norm=True):
        x = Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal', padding='same')(input_tensor)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def build_model(self):
        dropout = 0.1
        batch_norm = True
        shape1 = (2, 2)
        shape2 = (2, 2)
        shape3 = (2, 2)
        shape4 = (2, 2)

        filt1 = 128
        filt2 = 64
        filt3 = 1
        final_image_shape = [128, 128, 1]

        inp_img = Input(shape=final_image_shape, name='input')

        c1 = self.conv2d_block(inp_img, filt1, kernel_size=3, batch_norm=batch_norm)
        p1 = MaxPooling2D(shape1)(c1)
        p1 = Dropout(dropout)(p1)

        c2 = self.conv2d_block(p1, filt2, kernel_size=3, batch_norm=batch_norm)
        p2 = MaxPooling2D(shape2)(c2)
        p2 = Dropout(dropout)(p2)

        c33 = self.conv2d_block(p2, 32, kernel_size=3, batch_norm=batch_norm)
        c34 = MaxPooling2D(shape3)(c33)
        c35 = Dropout(dropout)(c34)

        encode_output = self.conv2d_block(c35, 1, kernel_size=2)

        # decoder:
        c3 = self.conv2d_block(encode_output, 32, kernel_size=3, batch_norm=batch_norm)
        c44 = UpSampling2D(shape3)(c3)
        c44 = Dropout(dropout)(c44)

        u4 = self.conv2d_block(c44, filt2, kernel_size=3, batch_norm=batch_norm)
        c4 = UpSampling2D(shape2)(u4)
        c4 = Dropout(dropout)(c4)

        u5 = self.conv2d_block(c4, filt1, kernel_size=3, batch_norm=batch_norm)
        c5 = UpSampling2D(shape1)(u5)
        c5 = Dropout(dropout)(c5)
        output1 = Conv2D(1, (1, 1), activation='sigmoid')(c5)
        i_x_i = Model(inputs=inp_img, outputs=output1)
        i_x = Model(inputs=inp_img, outputs=encode_output)

        # x squeeze
        x1 = self.conv2d_block(encode_output, 5, kernel_size=3, batch_norm=batch_norm)
        x2 = MaxPooling2D(shape3)(x1)
        x3 = self.conv2d_block(x2, 3, kernel_size=3, batch_norm=batch_norm)
        x4 = MaxPooling2D(shape3)(x3)
        x5 = self.conv2d_block(x4, 1, kernel_size=3, batch_norm=batch_norm)
        #squeze_x = Model(inputs=inp_img, outputs=x5)
        x = Flatten()(x5)

        # build x_j part
        
        j_out = layers.Dense(1, activation='relu')(x)
        i_x_j = Model(inputs=inp_img, outputs=j_out)
        x_j = Model(inputs = encode_output ,outputs=j_out)

        return i_x_i, i_x_j


    def save_latent_space(self, i_x, train_image, test_image):
        train_latent_space = i_x.predict(train_image)
        train_latent_space = train_latent_space.reshape((len(train_image),16,16))
        np.save(self.save_folder + 'train_latent_space', np.asarray(train_latent_space))
        test_latent_space = i_x.predict(test_image)
        test_latent_space = test_latent_space.reshape((len(test_image),16,16))
        np.save(self.save_folder + 'test_latent_space', np.asarray(test_latent_space))

    def modify_and_decode_random_images(self, i_x, x_i, x_j, random_images):
        train_latent_space = i_x.predict(random_images)
        predict_j_old = x_j.predict(train_latent_space)
        for modification_index, modification_factor in enumerate(np.arange(0.1, 1.1, 0.1), 1): 
            modified_latent_space = train_latent_space + np.random.normal(0, modification_factor, train_latent_space.shape)
            decoded_images = x_i.predict(modified_latent_space)
            predict_j_new = x_j.predict(modified_latent_space)
            
            # plot here
            self.plot_images(random_images, train_latent_space, modified_latent_space, decoded_images, predict_j_old, predict_j_new, modification_index)

    def plot_images(self, train_image, train_latent_space, modified_latent_space, decoded_images, predict_j_old, predict_j_new, modification_index):
        num_rows = train_image.shape[0]

        fig, axes = plt.subplots(num_rows, 4, figsize=(10, num_rows * 2))

        for i in range(num_rows):
            # Plot original image
            axes[i, 0].imshow(train_image[i].squeeze(), cmap='gray')
            axes[i, 0].set_title(f'Original image\nLabel: {predict_j_old[i]}')
            axes[i, 0].axis('off')

            # Plot original latent space
            axes[i, 1].imshow(train_latent_space[i].reshape(16, 16), cmap='gray')
            axes[i, 1].set_title('Original Latent Space')
            axes[i, 1].axis('off')

            # Plot new latent space
            axes[i, 2].imshow(modified_latent_space[i].reshape(16, 16), cmap='gray')
            axes[i, 2].set_title('New Latent Space')
            axes[i, 2].axis('off')

            # Plot new images
            axes[i, 3].imshow(decoded_images[i].squeeze(), cmap='gray')
            axes[i, 3].set_title(f'New Images\nLabel: {predict_j_new[i]}')
            axes[i, 3].axis('off')

        plt.tight_layout()
        plt.savefig(self.save_folder + f'random_noise_img_mod_{modification_index}.png')
        plt.show()



def extract_submodels(i_x_i, i_x_j):
    # Split i_x_i model into i_x and x_i models
    i_x_input = Input(shape=(128, 128, 1), name='input_img')
    x = i_x_input
    for layer in i_x_i.layers[:i_x_i.layers.index(i_x_i.get_layer("activation_3")) + 1]:
        x = layer(x)
    i_x = Model(inputs=i_x_input, outputs=x, name="i_x_model")

    x_i_input = Input(shape=(16, 16, 1), name='latent_input')
    x = x_i_input
    for layer in i_x_i.layers[i_x_i.layers.index(i_x_i.get_layer("conv2d_4")):]:
        x = layer(x)
    x_i = Model(inputs=x_i_input, outputs=x, name="x_i_model")

    # Split i_x_j model into x_j model
    x_j_input = Input(shape=(16, 16, 1), name='latent_input')
    x = x_j_input
    for layer in i_x_j.layers[i_x_j.layers.index(i_x_j.get_layer("conv2d_8")):]:
        x = layer(x)
    x_j = Model(inputs=x_j_input, outputs=x, name="x_j_model")

    return i_x, x_j, x_i

if __name__ == '__main__':
    
    bella_model = BellaModel()
    print('start load data')
    data, current, ff = bella_model.load_data()
    print('start split data')
    train_image, test_image, train_current, test_current, train_ff, test_ff = train_test_split(data, current, ff, test_size=0.2, random_state=57)

    #bella_model.save_train_test_data(train_image, test_image, train_current, test_current, train_ff, test_ff)
    print('start define model')
    i_x_i, i_x_j  = bella_model.build_model()

    auto_folder = '/work/baskargroup/bella/data_efficient/models/autoencoder/'
    i_x_i_checkpoint_path_old =  auto_folder + "i_x_i_00014.ckpt"
    print('start  auto load weight')
    i_x_i.load_weights(i_x_i_checkpoint_path_old)
    print('finish load model')

    print('start  i_x_j load weight')
    @tf.function
    def w_mse(y_true,y_pred):
        return kb.mean(kb.sum(y_true*(y_true-y_pred)**2))*0.008
    i_x_j_path = "/work/baskargroup/bella/keras_3_models/data_efficient/models/without_g/100%/model_i_x_j.h5"
    i_x_j = i_x_j = load_model(i_x_j_path, custom_objects={'w_mse': w_mse})
    print('finish i_x_j model')

    print('for i_x_i ')
    for layer in i_x_i.layers:
        print(layer.name, layer.output_shape)
    print('for i_x_j ')
    for layer in i_x_j.layers:
        print(layer.name, layer.output_shape)

    #extract sub models
    print('start  extract_submodels')
    i_x, x_j, x_i = extract_submodels(i_x_i, i_x_j)
    print('finish extract_submodels')
    print('for i_x ')
    for layer in i_x.layers:
        if not isinstance(layer, InputLayer):
            print(layer.name, layer.output_shape)
    print('for x_j ')
    for layer in x_j.layers:
        if not isinstance(layer, InputLayer):
            print(layer.name, layer.output_shape)   
    print('for x_i ')
    for layer in x_i.layers:
        if not isinstance(layer, InputLayer):
            print(layer.name, layer.output_shape)

    #bella_model.save_latent_space(i_x, train_image, test_image)

    # Add these lines after saving the latent space
    #train_latent_space = np.load(bella_model.save_folder + 'train_latent_space.npy')
    # random 5 images 
    num_images = train_image.shape[0]
    num_random_images = 5

    random_indices = np.random.randint(0, num_images, size=num_random_images)
    random_images = train_image[random_indices]

    bella_model.modify_and_decode_random_images(i_x,x_i,x_j, random_images)

    # train the ppo
    print('start training PPO')
    ppo_agent = train_ppo_agent(i_x, x_i, x_j, train_images, max_iterations=100, modification_factor=0.1, total_timesteps=50000)
    print('finish train ppo')



