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
            obs, _, done, _, info = env.step(action)
            # Reward.append(reward)
    print("=====DONE=====")
    # return Reward


if __name__ == "__main__":
    task, algo_name, lr = my_config['run_id'], my_config["algorithm"].__name__, my_config["learning_rate"]
    model_path = f"models/{task}/exp/3e-05_750"  # Change path to load different models
    
    env = gym.make('vlmHuman-v0', 
                   healthy_z_range=my_config["healthy_z_range"],
                   actionText=[
                            #    "Squatting down with knees bent and arms extended in front", 
                            #    "Standing back up with legs straight and arms at the sides"
                               "Squatting down with knees bent and hips lowered",
                               "Return to standing position with legs straight and hips raised",
                            ],
                   camera_config=DEFAULT_CAMERA_CONFIG,
                   max_episode_steps = my_config["max_episode_steps"],
                #    vlm_model_name=my_config["vlm_model_name"],
                #    vlm_model_version=my_config["vlm_model_version"],
                   render_mode="human",
                )

    ### Load model with SB3
    model = my_config["algorithm"].load(model_path)
    
    eval_num = 4
    evaluation(env, model, eval_num)

    # print("Avg_reward:  ", np.round(np.sum(Reward)/eval_num, 2))
    