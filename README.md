# ABCS Flocking: A Leader-Follower Collision Free UAV Flocking System with Multi-Agent Reinforcement Learning


> This is the **Attention Based Cucker-Smale Flocking algorithm**  implementation on [Multi-Agent Particle Environment(MPE)](https://github.com/openai/multiagent-particle-envs), the corresponding paper is [ABCS Flocking: A Leader-Follower Collision Free UAV Flocking System with Multi-Agent Reinforcement Learning](submitted) 

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

