import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np

env = gym.make("CartPole-v1")

model = PPO.load("data/cartpole_ppo")

obs_list = []
act_list = []
next_obs_list = []
done_list = []

obs, _ = env.reset()

for _ in range(10000):   # number of steps to collect
    action, _ = model.predict(obs)

    next_obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    obs_list.append(obs)
    act_list.append(action)
    next_obs_list.append(next_obs)
    done_list.append(done)

    obs = next_obs

    if done:
        obs, _ = env.reset()

print(len(obs_list))

obs_array = np.array(obs_list)
act_array = np.array(act_list)
next_obs_array = np.array(next_obs_list)
done_array = np.array(done_list)

np.savez(
    "data/expert_cartpole_data.npz",
    obs=obs_array,
    acts=act_array,
    next_obs=next_obs_array,
    dones=done_array
)

