# Installation
1. create conda env
```
conda create -n ENVNAME
```
2. install required dependencies
  * VLM-related: follow the instructions from [LAVIS](https://github.com/salesforce/LAVIS)
  ```
  conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
  pip install salesforce-lavis
  ```
> [!NOTE]
> Replace `cudatoolkit=11.0` to match your machine's cuda version

  * Others
  ```
  pip install stable-baselines3 sb3-contrib gym gym[mujoco]
  ```
# Usage
- `train.py` is used to train and save a robot motion policy
- `eval.py` is used to evaluate the trained policy by rendering
- `vlmReward.py` is used to get the similarity score. Also can be used to evaluate the score from a specific text and an input video
- `envs/myHumannoidEnv.py` is a customized gym env with an additional VLM reward
- `envs/humanoid_textured.xml` refers to [vlmrm](https://github.com/AlignmentResearch/vlmrm) which can modify texture

