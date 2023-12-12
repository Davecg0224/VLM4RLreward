import warnings
import gymnasium as gym
from gymnasium.envs.registration import register

# import wandb
# from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.utils import constant_fn, get_linear_fn
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from sb3_contrib import TQC
from sb3_contrib.tqc.policies import MlpPolicy, TQCPolicy

import numpy as np

register(
    id='vlmHuman-v0',
    entry_point='envs:VLMHumanoidEnv',
    # max_episode_steps = 100,
)

# Set camera configuration for the viewpoint
DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 3.5,
    "lookat": np.array((0.0, 0.0, 1.0)),
    "elevation": -10.0,
    "azimuth": 135.0,
}

# Set hyper params (configurations) for training
my_config = {
    "run_id": "squating",

    "algorithm": TQC,
    "policy_network": MlpPolicy,
    "save_path": "models",

    "learning_rate": 1e-5,
    "gamma": 0.9,
    "entropy_coef": 'auto',
    "learning_starts": 1000,

    "epoch_num": 2000,
    "timesteps_per_epoch": 500,
    "max_episode_steps": 100,
    "eval_episode_num": 10,
    "eval_freq": 10,

    "healthy_z_range": (0.2, 2.0),
    "vlm_model_name": "blip2_image_text_matching",
    "vlm_model_version": "pretrain",
}

def make_env():
    env = gym.make('vlmHuman-v0', 
                   healthy_z_range=my_config["healthy_z_range"],
                   actionText=[
                                "Squatting down with knees bent and hips lowered",
                                "Standing up with legs straight and hips raised",
                            ],
                   camera_config=DEFAULT_CAMERA_CONFIG,
                   max_episode_steps = my_config["max_episode_steps"],
                   vlm_model_name=my_config["vlm_model_name"],
                   vlm_model_version=my_config["vlm_model_version"],
                #    render_mode="human",
                )
    return env

def train(env, model, config):

    current_best = 0

    for epoch in range(config["epoch_num"]):

        ### Train agent using SB3
        # Uncomment to enable wandb logging
        model.learn(
            total_timesteps=config["timesteps_per_epoch"],
            reset_num_timesteps=False,
            # callback=WandbCallback(
            #     gradient_save_freq=100,
            #     verbose=2,
            # ),
        )

        ### Evaluation
        if epoch % config["eval_freq"] == 0 or epoch == config["epoch_num"] - 1:
            print(config["run_id"])
            print("Epoch: ", epoch)
            avg_reward = 0
            for seed in range(config["eval_episode_num"]):
                done = False

                # Set seed using old Gym API
                env.seed(seed)
                obs = env.reset()

                # Interact with env using old Gym API
                while not done:
                    action, _state = model.predict(obs, deterministic=True)
                    obs, reward, done, info = env.step(action)

                    avg_reward  += reward/config["eval_episode_num"]
            
            print("Avg_reward: ", np.round(avg_reward, 2) )

            
            # wandb.log(
            #     {"avg_reward": avg_reward}
            # )

            ## Save best model
            if current_best < avg_reward:
                print("Saving Model")
                current_best = avg_reward
                save_path = config["save_path"]
                algo_name, lr = config["algorithm"].__name__, config["learning_rate"]
                model.save(f"{save_path}/{my_config['run_id']}/{algo_name}/2stage_{lr}_{epoch}")

            print("---------------")

if __name__ == "__main__":
    # print(my_config["run_id"])
    # Create wandb session (Uncomment to enable wandb logging)
    # run = wandb.init(
    #     project="rlfp",
    #     config=my_config,
    #     sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    #     id=my_config["run_id"]
    # )

    env = DummyVecEnv([make_env])

    # Create model from loaded config and train
    # Note: Set verbose to 0 if you don't want info messages
    model = my_config["algorithm"](
        my_config["policy_network"], 
        env,
        my_config["learning_rate"],
        learning_starts = my_config["learning_starts"],
        # use_rms_prop=False,
        use_sde=True, 
        gamma = my_config["gamma"],
        ent_coef = my_config["entropy_coef"],
        use_sde_at_warmup=True,
        verbose=0,
        # tensorboard_log=my_config["run_id"],
        device='auto'
    )
    train(env, model, my_config)