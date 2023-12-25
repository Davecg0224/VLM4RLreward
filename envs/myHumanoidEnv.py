import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
from vlmReward import VLM

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

def mass_center(model, data):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()


class VLMHumanoidEnv(MujocoEnv, utils.EzPickle):    

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 67,
    }

    def __init__(
        self,
        forward_reward_weight=1.25,
        ctrl_cost_weight=0.1,
        healthy_reward=5.0,
        terminate_when_unhealthy=True,
        max_episode_steps=1000,
        healthy_z_range=(1.0, 2.0),
        reset_noise_scale=1e-2,
        exclude_current_positions_from_observation=True,
        actionText = ["walking"],
        camera_config=None,
        vlm_model_name = 'clip_feature_extractor',
        vlm_model_version = "ViT-L-14",
        **kwargs
    ):
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            actionText,
            **kwargs
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._stepCount = 0
        self._prev_w_score = 0
        self._prev_score = np.zeros(2)
        self.max_episode_steps = max_episode_steps

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        self.actionText = actionText
        self.camera_config = camera_config
        self.vlm_model_name = vlm_model_name
        self.vlm_model_version = vlm_model_version

        if exclude_current_positions_from_observation:
            observation_space = Box(
                # low=-np.inf, high=np.inf, shape=(317,), dtype=np.float64
                # low=-np.inf, high=np.inf, shape=(320,), dtype=np.float64
                low=-np.inf, high=np.inf, shape=(376,), dtype=np.float64
            )
        else:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(378,), dtype=np.float64
            )

        MujocoEnv.__init__(
            self, 
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "humanoid_textured.xml"), 
            5, 
            observation_space=observation_space,
            default_camera_config=self.camera_config,
            **kwargs
        )

        # init vlm
        if self.render_mode != "human":
            self.vlm = VLM(model_name=self.vlm_model_name, 
                        model_version=self.vlm_model_version,
                        actionText=self.actionText,
                        use_itm=True,
                        )
        
    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(self.data.ctrl))
        return control_cost

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.data.qpos[2] < max_z

        return is_healthy

    @property
    def terminated(self):
        terminated = (not self.is_healthy) if self._terminate_when_unhealthy else False
        return terminated

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        com_inertia = self.data.cinert.flat.copy()
        com_velocity = self.data.cvel.flat.copy()

        actuator_forces = self.data.qfrc_actuator.flat.copy()
        external_contact_forces = self.data.cfrc_ext.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        return np.concatenate(
            (
                position,
                velocity,
                com_inertia,
                com_velocity,
                actuator_forces,
                external_contact_forces,
            )
        )

    def step(self, action):
        self._stepCount += 1
        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()
        
        if self._stepCount > self.max_episode_steps:
            terminated = True
        else:
            terminated = self.terminated

        if not self.is_healthy:
            reward = -self._healthy_reward
            info = {
                'squating_score': 0,
                'standing_score': 0,
                'DW_score': 0,
                'WD_score': 0,
            }   
            return observation, reward, terminated, False, info

        if self.render_mode == "human":
            self.render()
            return observation, 0, terminated, False, {}

        # render image for VLM scoring
        img = self.mujoco_renderer.render(render_mode = "rgb_array", camera_id = -1)
        score = self.vlm.getScore(img)

        ## switch-based
        # selected_score = s1
        # # chanege to sec. stage while reaches half of the max_episode_steps
        # if self._stepCount >= self.max_episode_steps/2:
        #     # reset self._prev_score once
        #     # if self._stepCount == self.max_episode_steps/2:
        #     #     self._prev_score = 0
                
        #     selected_score = s2

        ## weighted baselines
        stepRatio = self._stepCount / self.max_episode_steps
        w_lin = 1 - stepRatio
        w_exp = np.exp(-stepRatio * 10)
        w_tanh = 1 - 0.5*(np.tanh(self._stepCount - self.max_episode_steps/2) + 1)
        w_cos = 1 - 0.5*(np.cos(2*np.pi*stepRatio) + 1)
        w = w_tanh

        ## delta-weighted based
        weighted_score = score @ np.array([w, (1-w)*2])
        delta_weighted_score = weighted_score - self._prev_w_score
        self._prev_w_score = weighted_score
        # reward = weighted_score - self._prev_score

        ## weighted-delta based
        weighted_delta_score = (score - self._prev_score) @ np.array([w, (1-w)])
        self._prev_score = score
        # reward = weighted_delta_score

        ## reward selection
        reward = weighted_score

        info = {
            'squating_score': score[0],
            'standing_score': score[1],
            'DW_score': delta_weighted_score,
            'WD_score': weighted_delta_score,
        }            

        return observation, reward, terminated, False, info

    def reset_model(self):
        self._stepCount = 0
        self._prev_w_score = 0
        self._prev_score = np.zeros(2)

        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in self.camera_config.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
