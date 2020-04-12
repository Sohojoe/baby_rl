# baby_rl
Reinforcement Learning Algorithms in PyTorch


Based on https://github.com/ShangtongZhang/DeepRL
 hash [5d0ad07](https://github.com/ShangtongZhang/DeepRL/commit/5d0ad07c7f2081123fddc4faf8db2aa09730e85b)


* Use Marathon Envs instead of Majoco
* removed dependency on OpenAI Baselines

Implemented algorithms:
* Twined Delayed DDPG (TD3)


# Dependency

* PyTorch v1.4.0
* Python 3.6, 3.5
* Core dependencies: pip install -e .
* Download MarathonEnvs binary
 * xxxxx show how ****


# Install
```
conda env create -f environment.yml
conda activate baby_rl
# manually install marathon-envs
pip install -r requirements.txt
pip install -e .
```

for tensorboard
```
tensorboard --logdir=tf_log
```


# Usage

examples.py contains examples for all the implemented algorithms


# References

* [Modularized Implementation of Deep RL Algorithms in PyTorch](https://github.com/ShangtongZhang/DeepRL)
