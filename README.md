# Installation
### Create the conda env with the dependencies
1. create conda env
```
conda create -n ENVNAME
```
2. install required dependencies
  * VLM-related: follow the instructions from [CLIP](https://github.com/openai/CLIP)
  ```
  conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
  pip install ftfy regex tqdm
  pip install git+https://github.com/openai/CLIP.git
  ```
> [!NOTE]
> Replace `cudatoolkit=11.0` to match your machine's cuda version

  * Others
  ```
  conda install -c conda-forge gym stable-baselines3
  conda install -c anaconda opencv
  ```
# Usage
- `train.py` is used to train and save a robot motion policy
- `eval.py` is used to evaluate the trained policy by rendering
- `vlmReward.py` is used to get the similarity score. Also can be used to evaluate the score from specific text and input video
- `envs/myHumannoidEnv.py` is a customized gym env with an additional VLM reward
- `envs/humanoid_textured.xml` refers to [vlmrm](https://github.com/AlignmentResearch/vlmrm) which can modify texture

