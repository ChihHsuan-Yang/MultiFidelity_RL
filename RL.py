import gym
import numpy as np
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

class LatentSpaceEnvironment(gym.Env):
    def __init__(self, i_x, x_i, x_j, train_images, max_iterations=100, modification_factor=0.1, step_size=0.1):
        super(LatentSpaceEnvironment, self).__init__()
        self.i_x = i_x
        self.x_i = x_i
        self.x_j = x_j
        self.train_images = train_images
        self.max_iterations = max_iterations
        self.modification_factor = modification_factor
        self.step_size = step_size

        self.train_latent_space = self._get_random_latent_space()

        self.n_actions = int((1 - (-1)) / step_size) + 1
        self.action_space = spaces.Box(low=-1, high=1, shape=(16, 16, 1), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(16, 16, 1), dtype=np.uint8)
        self.iterations = 0

    def _get_random_latent_space(self):
        random_index = np.random.randint(self.train_images.shape[0])
        random_image = self.train_images[random_index]
        return self.i_x.predict(np.array([random_image]))[0]

    def step(self, action):
        action_values = action * self.step_size
        self.train_latent_space += action_values * self.modification_factor
        decoded_images = self.x_i.predict(np.array([self.train_latent_space]))
        predict_j = self.x_j.predict(np.array([self.train_latent_space]))

        reward = np.mean(predict_j)
        self.iterations += 1
        done = self.iterations >= self.max_iterations
        info = {}
        return self.train_latent_space, reward, done, info

    def reset(self):
        self.iterations = 0
        self.train_latent_space = self._get_random_latent_space()
        return self.train_latent_space

    def render(self, mode='human'):
        pass


def train_ppo_agent(i_x, x_i, x_j, train_images, batch_size=64, modification_factor=0.1, total_timesteps=50000, save_path="/data/bella/data_efficient/coms590_rl/model/saved_rl_agent/"):
    env = LatentSpaceEnvironment(i_x, x_i, x_j, train_images, batch_size, modification_factor)
    env = Monitor(env)  # Add Monitor wrapper
    env = DummyVecEnv([lambda: env])

    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=save_path)  # Save the agent every 10000 steps

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    return model

def plot_training_loss(log_folder, title="PPO Training Loss"):
    results_plotter.plot_results([log_folder], total_timesteps=50000, results_per_file=None, title=title)
    plt.show()

def modify_image_with_ppo_agent(ppo_agent, i_x, x_i, modification_factor, input_image):
    # Convert input image to latent space representation
    latent_space = i_x.predict(np.expand_dims(input_image, axis=0))

    # Get action (modification) from the PPO agent
    action, _ = ppo_agent.predict(latent_space)

    # Apply the action to the latent space representation
    modified_latent_space = latent_space + action * modification_factor

    # Decode the modified latent space back to an image
    modified_image = x_i.predict(modified_latent_space)

    return modified_image.squeeze()


