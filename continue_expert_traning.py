import gymnasium as gym
from stable_baselines3 import PPO

# create environment
env = gym.make("CartPole-v1")

# load saved model
model = PPO.load("cartpole_ppo_100000", env=env)

# continue training
model.learn(total_timesteps=50000)

# save again
model.save("cartpole_ppo")