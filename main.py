import tensorflow as tf
import keras
print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)

from Model import *
from RL import *


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