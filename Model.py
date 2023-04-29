import numpy as np
import keras
from keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Dropout, UpSampling2D, Flatten, Dense, Input
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras import layers

class BellaModel:
    def __init__(self):
        self.save_folder = '/work/baskargroup/bella/data_efficient/coms590_rl/report/test/'
        self.path = '/work/baskarg/bella/data_efficient/data/'
    
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
        squeze_x = Model(inputs=inp_img, outputs=x5)

        # g input
        g_input = Input(shape=(3,), name='g_input')
        x = Flatten()(x5)

        # build x_g_j part
        j0 = layers.concatenate([x, g_input], axis=-1)
        j_out = layers.Dense(1, activation='relu')(j0)
        x_g_j = Model(inputs=[inp_img, g_input], outputs=[j_out])

        return i_x_i, i_x, squeze_x, x_g_j


    def save_latent_space(self, i_x, train_image, test_image):
        train_latent_space = i_x.predict(train_image)
        train_latent_space = train_latent_space.reshape((len(train_image),16,16))
        np.save(self.save_folder + 'train_latent_space', np.asarray(train_latent_space))
        test_latent_space = i_x.predict(test_image)
        test_latent_space = test_latent_space.reshape((len(test_image),16,16))
        np.save(self.save_folder + 'test_latent_space', np.asarray(test_latent_space))

    # Add this new method to the BellaModel class
    def modify_and_decode_latent_space(self, i_x_i, train_latent_space, modification_factor=0.1):
        modified_latent_space = train_latent_space + np.random.normal(0, modification_factor, train_latent_space.shape)
        decoded_images = i_x_i.predict(modified_latent_space)
        return decoded_images

if __name__ == '__main__':
    
    bella_model = BellaModel()
    print('start load data')
    data, current, ff = bella_model.load_data()
    print('start split data')
    train_image, test_image, train_current, test_current, train_ff, test_ff = train_test_split(data, current, ff, test_size=0.2, random_state=57)

    bella_model.save_train_test_data(train_image, test_image, train_current, test_current, train_ff, test_ff)
    print('start define model')
    i_x_i, i_x, squeze_x, x_g_j = bella_model.build_model()

    auto_folder = '/work/baskarg/bella/data_efficient/auto_encoder/'
    i_x_i_checkpoint_path_old =  auto_folder + "i_x_i_00014.ckpt"
    print('start load weight')
    i_x_i.load_weights(i_x_i_checkpoint_path_old)

    bella_model.save_latent_space(i_x, train_image, test_image)
    # Add these lines after saving the latent space
    train_latent_space = np.load(bella_model.save_folder + 'train_latent_space.npy')
    modified_decoded_images = bella_model.modify_and_decode_latent_space(i_x_i, train_latent_space)
    # Save the modified decoded images if needed
    np.save(bella_model.save_folder + 'modified_decoded_images', np.asarray(modified_decoded_images))

