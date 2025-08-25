# An invulnerable leader–follower collision-free unmanned aerial vehicle flocking system with attention-based Multi-Agent Reinforcement Learning


> This is the **Attention Based Cucker-Smale Flocking algorithm**  implementation on [Multi-Agent Particle Environment(MPE)](https://github.com/openai/multiagent-particle-envs), the corresponding paper is [An invulnerable leader–follower collision-free unmanned aerial vehicle flocking system with attention-based Multi-Agent Reinforcement Learning](https://doi.org/10.1016/j.engappai.2025.111797) 

The MADDPG part is come from: [MADDPG](https://gitee.com/ming_autumn/MADDPG-1?_from=gitee_search)


## Requirements

- python=3.6.5
- [Multi-Agent Particle Environment(MPE)](https://github.com/openai/multiagent-particle-envs)
- torch=1.1.0


**Or download the python environment directly: [LG-CS.zip](https://pan.baidu.com/s/1ODtPNWxLOWAHcw7ZDz2sWw)**

**Extract code: MARL**

## Complie Cython Code
Before running the code, please complie the environment code:

```shell
cd ./envs
python setup.py build_ext --inplace --force
```

## Training Agents
Running the main.py, the agents will learn from the flocking scenario：
```shell
python main.py --n-agents=5 --evaluate-episodes=256
```
If you want to adjust the parameters, please see the `./common/arguments.py` for more details.



## Testing Agents
```shell
python main.py --n-agents=5 --evaluate-episodes=10 --evaluate=True
```

## Display Results

After data collection:

```shell
python display.py
```

# Citation

If you use this code, please cite our paper:

[1] Yunxiao Guo, Dan Xu, Chang Wang, Jinxi Li, Han Long, An invulnerable leader–follower collision-free unmanned aerial vehicle flocking system with attention-based Multi-Agent Reinforcement Learning,Engineering Applications of Artificial Intelligence,
2025,160, Part C,111797,doi: 10.1016/j.engappai.2025.111797. 

Bibtex form:
```
@article{GUO2025111797,
author = {Yunxiao Guo and Dan Xu and Chang Wang and Jinxi Li and Han Long},
title = {An invulnerable leader–follower collision-free unmanned aerial vehicle flocking system with attention-based Multi-Agent Reinforcement Learning},
journal = {Engineering Applications of Artificial Intelligence},
volume = {160},
pages = {111797},
year = {2025},
issn = {0952-1976},
doi = {https://doi.org/10.1016/j.engappai.2025.111797},
url = {https://www.sciencedirect.com/science/article/pii/S0952197625017993}
}
```
