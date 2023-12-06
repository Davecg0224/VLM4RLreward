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
        healthy_z_range=(1.0, 2.0),
        reset_noise_scale=1e-2,
        exclude_current_positions_from_observation=True,
        actionText = ["walking"],
        step_penalty = -0.1,
        onlyVLMreward = True,
        camera_config=None,
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
            step_penalty,
            onlyVLMreward,
            **kwargs
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._step_penalty = step_penalty

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        self.actionText = actionText
        self._onlyVLMreward = onlyVLMreward
        self.camera_config = camera_config
        self._step_counter = 0

        if exclude_current_positions_from_observation:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(317,), dtype=np.float64
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
        self.vlm = VLM(model_name='blip_feature_extractor', 
                       model_version='base',
                       actionText=self.actionText)

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
        self._step_counter += 1
        xy_position_before = mass_center(self.model, self.data)
        self.do_simulation(action, self.frame_skip)

        # render image for VLM reward
        img = self.mujoco_renderer.render(render_mode = "rgb_array", camera_id = 0)

        if self._onlyVLMreward:
            reward = self.vlm.getScore(img) - self._step_penalty * self._step_counter
            info = {}

        else:
            xy_position_after = mass_center(self.model, self.data)
            xy_velocity = (xy_position_after - xy_position_before) / self.dt
            x_velocity, y_velocity = xy_velocity

            ctrl_cost = self.control_cost(action)

            forward_reward = self._forward_reward_weight * x_velocity
            healthy_reward = self.healthy_reward

            reward = forward_reward + healthy_reward - ctrl_cost
            reward += self.vlm.getScore(img)

            info = {
            "reward_linvel": forward_reward,
            "reward_quadctrl": -ctrl_cost,
            "reward_alive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
            "vlm_reward": self.vlm.getScore(img),
            }

        observation = self._get_obs()
        terminated = self.terminated

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def reset_model(self):
        self._step_counter = 0
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
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
