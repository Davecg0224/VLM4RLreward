import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO, A2C, SAC, TD3
from sb3_contrib import TQC

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
register(
    id='vlmHuman-v0',
    entry_point='envs:VLMHumanoidEnv',
)

def evaluation(env, model, render_last, eval_num=100):
    Reward = []

    ### Run eval_num times rollouts
    for seed in range(eval_num+10):
        done = False
        # Set seed and reset env using Gymnasium API
        obs, info = env.reset(seed=seed)

        while not done:
            # Interact with env using Gymnasium API
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            Reward.append(reward)

    return Reward


if __name__ == "__main__":
    model_path = "models/TQC/3e-05_80"  # Change path name to load different modelsE
    env = gym.make('vlmHuman-v0', render_mode="human")

    ### Load model with SB3
    modelName = TQC
    model = modelName.load(model_path)
    
    eval_num = 10
    Reward = evaluation(env, model, True, eval_num)

    print("Avg_reward:  ", np.round(np.sum(Reward)/eval_num, 2))
    