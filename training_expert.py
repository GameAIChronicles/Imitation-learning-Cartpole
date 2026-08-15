import gymnasium as gym
from stable_baselines3 import PPO
import time
# Create environment
env = gym.make("CartPole-v1", render_mode='human')

# Create RL model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1
)
time.sleep(10)
# Train model
model.learn(total_timesteps=100000)

# Save model
model.save("cartpole_ppo")