# Imitation-learning-Cartpole
This repository provides a complete pipeline for training a CartPole-v1 agent using a combination of Reinforcement Learning and Imitation Learning. It includes scripts to train an expert PPO model, extract 10,000 steps of expert trajectory data, and train a new policy using Behavioral Cloning (BC) before fine-tuning it further with PPO.

# CartPole Imitation Learning & PPO Pipeline

This project demonstrates how to train a Gymnasium `CartPole-v1` agent by bootstrapping a Reinforcement Learning model with Behavioral Cloning (Imitation Learning). It uses `stable_baselines3` for PPO implementations and the `imitation` library for Behavioral Cloning[cite: 1, 5].

## 🚀 Project Workflow

1. **Train an Expert Model:** The `training_expert.py` script trains an initial PPO model on `CartPole-v1` from scratch for 100,000 timesteps and saves the model[cite: 5]. 
2. **Fine-tune the Expert (Optional):** The `continue_expert_traning.py` script allows you to load an existing model and train it for an additional 50,000 timesteps[cite: 2].
3. **Collect Expert Data:** The `expert_data_sample.py` script loads the trained expert PPO model and runs it for 10,000 steps[cite: 4]. It records the observations, actions, next observations, and termination states, saving them to `data/expert_cartpole_data.npz`[cite: 4].
4. **Train via Behavioral Cloning:** The `bc_imitation.py` script loads the `.npz` dataset and converts it into Trajectory objects[cite: 1]. It trains a Behavioral Cloning (BC) policy for 10 epochs[cite: 1].
5. **RL Fine-tuning:** After BC training, `bc_imitation.py` transfers the learned weights into a new PPO model and continues training with reinforcement learning for another 1,000 timesteps, saving the final agent as `cartpole_imitation_rl`[cite: 1].

## 📁 Repository Structure

*   `data/`: Directory containing the saved expert PPO models (`cartpole_ppo.zip`) and the generated demonstration dataset (`expert_cartpole_data.npz`).
*   `training_expert.py`: Script to train the initial PPO expert[cite: 5].
*   `continue_expert_traning.py`: Script to resume training on an existing PPO model[cite: 2].
*   `expert_data_sample.py`: Script to collect observation and action data from the expert model[cite: 4].
*   `bc_imitation.py`: Main script to train the Behavioral Cloning agent and fine-tune it with PPO[cite: 1].
*   `env.py` / `test.py`: Scripts used to render the environment and run basic random action sampling tests[cite: 3].

## 🎥 Video Walkthrough
Watch the full breakdown and explanation of this project on YouTube:
[![CartPole Behavioral Cloning Tutorial](https://img.youtube.com/vi/MNw-busuMHY/maxresdefault.jpg)](https://youtu.be/MNw-busuMHY)
