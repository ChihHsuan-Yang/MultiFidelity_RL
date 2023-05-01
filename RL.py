import gym
import numpy as np
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
#comment
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
        self.action_space = spaces.MultiDiscrete([self.n_actions] * self.train_latent_space.shape[1])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.train_latent_space.shape[1],), dtype=np.float32)

        self.iterations = 0

    def _get_random_latent_space(self):
        random_index = np.random.randint(self.train_images.shape[0])
        random_image = self.train_images[random_index]
        return self.i_x.predict(np.array([random_image]))[0]

    def step(self, action):
        action_values = (action - (self.n_actions - 1) // 2) * self.step_size
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


def train_ppo_agent(i_x, x_i, x_j, train_images, batch_size, modification_factor=0.1, total_timesteps=50000):
    env = LatentSpaceEnvironment(i_x, x_i, x_j, train_images, batch_size, modification_factor)
    env = DummyVecEnv([lambda: env])

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)

    return model

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


