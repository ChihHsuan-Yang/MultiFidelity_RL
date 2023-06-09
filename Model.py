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
import seaborn as sns
from RL import *

class BellaModel:
    def __init__(self):
        self.save_folder = '/data/bella/data_efficient/coms590_rl/report/test_rl/'
        self.path = '/data/bella/data_efficient/data/'
    
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

def modify_and_compare_images(model, i_x, x_i, x_j, ppo_agent, test_images, modification_factor):
    num_images = test_images.shape[0]

    # Get the latent space of the test images
    latent_space_test = i_x.predict(test_images)

    # Add random noise to the latent space
    noisy_latent_space = latent_space_test + np.random.normal(0, modification_factor/3, latent_space_test.shape)
    print('noisy_latent_space shape',noisy_latent_space.shape)
    # Get RL-modified latent spaces using the PPO agent
    print("start geting ls from agent")
    rl_latent_space = np.array([modify_ls_with_ppo_agent(ppo_agent, i_x, x_i, modification_factor, test_images[i]) for i in range(num_images)])
    print('pass the agent')
    print('rl_latent_space shape',rl_latent_space.shape)
    rl_latent_space = np.reshape(rl_latent_space, noisy_latent_space.shape)
    # Decode the latent spaces back to images
    print('pass the agent and start x_i')
    initial_images = x_i.predict(latent_space_test)
    print('get initial_images')
    noise_images = x_i.predict(noisy_latent_space)
    print('get noise_images')
    rl_images = x_i.predict(rl_latent_space)
    print('get rl_images')

    # Get predicted labels
    initial_labels = x_j.predict(latent_space_test)
    print('get initial_labels')
    noise_labels = x_j.predict(noisy_latent_space)
    print('get noise_labels')
    rl_labels = x_j.predict(rl_latent_space)
    print('get rl_labels')

    # Plot the images and latent spaces
    fig, axes = plt.subplots(num_images, 6, figsize=(15, num_images * 2))

    for i in range(num_images):
        # Initial image and latent space
        axes[i, 0].imshow(initial_images[i].squeeze(), cmap='gray')
        axes[i, 0].set_title('Initial Reconstructed image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(latent_space_test[i].reshape(16, 16), cmap='gray')
        axes[i, 1].set_title(f'Initial Latent Space\nLabel: {initial_labels[i]}')
        axes[i, 1].axis('off')

        # Noisy image and latent space
        axes[i, 2].imshow(noise_images[i].squeeze(), cmap='gray')
        axes[i, 2].set_title('Noisy reconstructed image')
        axes[i, 2].axis('off')

        axes[i, 3].imshow(noisy_latent_space[i].reshape(16, 16), cmap='gray')
        axes[i, 3].set_title(f'Noisy Latent Space\nLabel: {noise_labels[i]}')
        axes[i, 3].axis('off')

        # RL image and latent space
        axes[i, 4].imshow(rl_images[i].squeeze(), cmap='gray')
        axes[i, 4].set_title('RL reconstructed image')
        axes[i, 4].axis('off')

        axes[i, 5].imshow(rl_latent_space[i].reshape(16, 16), cmap='gray')
        axes[i, 5].set_title(f'RL Latent Space\nLabel: {rl_labels[i]}')
        axes[i, 5].axis('off')

    plt.tight_layout()
    plt.savefig(model.save_folder + f'comaparation_{modification_factor}.png')
    plt.show()


def analyze_images_with_noise_and_rl(model, i_x, x_i, x_j, ppo_agent, test_images, modification_factor):
    num_images = test_images.shape[0]

    # Get the latent space of the test images
    latent_space_test = i_x.predict(test_images)

    # Add random noise to the latent space
    random_values = np.random.uniform(-modification_factor, modification_factor, latent_space_test.shape)
    noisy_latent_space = latent_space_test + random_values
    #noisy_latent_space = latent_space_test + np.random.normal(0, modification_factor/3, latent_space_test.shape)

    # Get RL-modified latent spaces using the PPO agent
    rl_latent_space = np.array([modify_ls_with_ppo_agent(ppo_agent, i_x, x_i, modification_factor, test_images[i]) for i in range(num_images)])
    rl_latent_space = np.reshape(rl_latent_space, noisy_latent_space.shape)
    # Get predicted labels
    initial_labels = x_j.predict(latent_space_test)
    noise_labels = x_j.predict(noisy_latent_space)
    rl_labels = x_j.predict(rl_latent_space)

    # Plot scatter plots
    plt.figure(figsize=(10, 5))
    plt.scatter(initial_labels, noise_labels, color='royalblue', label='Noise', alpha=0.5)
    plt.scatter(initial_labels, rl_labels, color='limegreen', label='RL', alpha=0.5)
    plt.xlabel('Initial Label')
    plt.ylabel('Modified Label')
    plt.legend()
    plt.title('Scatter plot of Noise vs RL modifications')
    plt.savefig(model.save_folder + 'scatter_plot.png')
    plt.show()

    # Plot box plot
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=[noise_labels - initial_labels, rl_labels - initial_labels], orient="h", palette="Set2")
    plt.yticks([0, 1], ['Noise', 'RL'])
    plt.xlabel('Difference (Modified - Initial)')
    plt.title('Box plot of differences between RL and Noise modifications')
    plt.savefig(model.save_folder + f'box_plot_difference{modification_factor}.png')
    plt.show()

if __name__ == '__main__':
    
    bella_model = BellaModel()
    print('start load data')
    data, current, ff = bella_model.load_data()
    print('start split data')
    train_image, test_image, train_current, test_current, train_ff, test_ff = train_test_split(data, current, ff, test_size=0.2, random_state=57)

    #bella_model.save_train_test_data(train_image, test_image, train_current, test_current, train_ff, test_ff)
    print('start define model')
    i_x_i, i_x_j  = bella_model.build_model()

    auto_folder = '/data/bella/data_efficient/models/autoencoder/'
    i_x_i_checkpoint_path_old =  auto_folder + "i_x_i_00014.ckpt"
    print('start  auto load weight')
    i_x_i.load_weights(i_x_i_checkpoint_path_old)
    print('finish load model')

    print('start  i_x_j load weight')
    @tf.function
    def w_mse(y_true,y_pred):
        return kb.mean(kb.sum(y_true*(y_true-y_pred)**2))*0.008
    i_x_j_path = "/data/bella/data_efficient/models/without_g/100%/model_i_x_j.h5"
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
    # Train the PPO agent
    
    log_folder = "/data/bella/data_efficient/coms590_rl/report/logs/"
    ppo_agent = train_ppo_agent(i_x, x_i, x_j, train_image, modification_factor=0.5, total_timesteps=150000, save_path=log_folder)
    
    #model_file_path = "/data/bella/data_efficient/coms590_rl/report/logs_03/"+"rl_model_151500_steps.zip"
    #ppo_agent = PPO.load(model_file_path)
    print('finish load rl agent')
    # Plot the training loss
    #plot_training_loss(log_folder)


    num_images = test_image.shape[0]
    num_random_images = 10
    random_indices = np.random.randint(0, num_images, size=num_random_images)
    random_test_images = test_image[random_indices]
    print('start run the comparison')

    modification_factor = 0.5
    modify_and_compare_images(bella_model, i_x, x_i, x_j, ppo_agent, random_test_images, modification_factor = modification_factor)
    print('modify_and_compare_images')
    analyze_images_with_noise_and_rl(bella_model, i_x, x_i, x_j, ppo_agent, test_image, modification_factor)

