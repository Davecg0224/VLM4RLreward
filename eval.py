import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO, A2C, SAC, TD3
from sb3_contrib import TQC
from train import my_config, DEFAULT_CAMERA_CONFIG

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def evaluation(env, model, eval_num=100):
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
    task, algo_name, lr = my_config['run_id'], my_config["algorithm"].__name__, my_config["learning_rate"]
    model_path = f"models/{task}/{algo_name}/3e-05_0"  # Change path to load different models
    
    env = gym.make('vlmHuman-v0', 
                   healthy_z_range=(0.5, 2.0), 
                   actionText=["Squatting down with knees bent and arms extended in front"], 
                   camera_config=DEFAULT_CAMERA_CONFIG,
                   render_mode="human",
                )

    ### Load model with SB3
    model = my_config["algorithm"].load(model_path)
    
    eval_num = 10
    Reward = evaluation(env, model, eval_num)

    print("Avg_reward:  ", np.round(np.sum(Reward)/eval_num, 2))
    