import numpy as np
from imitation.algorithms.bc import BC
from imitation.data.types import Trajectory
import gymnasium as gym
from stable_baselines3 import PPO
import time


env = gym.make("CartPole-v1", render_mode='human')
policy = PPO(
    "MlpPolicy",
    env,
    verbose=1).policy


data = np.load("data/expert_cartpole_data.npz")

obs = data["obs"]
acts = data["acts"]
dones = data["dones"]
next_obs = data["next_obs"]


trajectories = []

current_obs = []
current_acts = []
for i in range(len(obs)):
    current_obs.append(obs[i])
    current_acts.append(acts[i])

    if dones[i]:
        current_obs.append(next_obs[i])

        traj = Trajectory(
            obs=np.array(current_obs),
            acts=np.array(current_acts),
            infos=None,
            terminal = True)
        trajectories.append(traj)

        current_obs = []
        current_acts = []


rng = np.random.default_rng()
bc_trainer = BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=trajectories,
    policy=policy,
    rng=rng)


initial = time.time()
bc_trainer.train(n_epochs=10)
final = time.time()
print("Training time: ", final - initial)

# get trained BC policy
trained_policy = bc_trainer.policy

# create RL model using the BC policy
rl_model = PPO(
    policy='MlpPolicy',
    env=env,
    verbose=1
)

rl_model.policy.load_state_dict(bc_trainer.policy.state_dict())

initial = time.time()
# continue training with RL
rl_model.learn(
    total_timesteps=1000,
    reset_num_timesteps=False)
final = time.time()
print("Training time: ", final - initial)

# save final model
rl_model.save("cartpole_imitation_rl")

