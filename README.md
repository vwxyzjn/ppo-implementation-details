# The 37 Implementation Details of Proximal Policy Optimization

This repo contains the source code for the blog post *The 37 Implementation Details of Proximal Policy Optimization*

* Blog post url: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
* Tracked Weights and Biases experiments: https://wandb.ai/vwxyzjn/ppo-details

If you like this repo, consider checking out CleanRL (https://github.com/vwxyzjn/cleanrl), the RL library that we used to build this repo.


## Get started

Prerequisites:
* Python 3.8+
* [Poetry](https://python-poetry.org)

Install dependencies:
```
poetry install
```

Train agents:
```
poetry run python ppo.py
```

Train agents with experiment tracking:
```
poetry run python ppo.py --track --capture-video
```

### Atari
Install dependencies:
```
poetry install -E atari
```
Train agents:
```
poetry run python ppo_atari.py
```
Train agents with experiment tracking:
```
poetry run python ppo_atari.py --track --capture-video
```


### Pybullet
Install dependencies:
```
poetry install -E pybullet
```
Train agents:
```
poetry run python ppo_continuous_action.py
```
Train agents with experiment tracking:
```
poetry run python ppo_continuous_action.py --track --capture-video
```


### Gym-microrts (MultiDiscrete)

Install dependencies:
```
poetry install -E gym-microrts
```
Train agents:
```
poetry run python ppo_multidiscrete.py
```
Train agents with experiment tracking:
```
poetry run python ppo_multidiscrete.py --track --capture-video
```
Train agents with invalid action masking:
```
poetry run python ppo_multidiscrete_mask.py
```
Train agents with invalid action masking and experiment tracking:
```
poetry run python ppo_multidiscrete_mask.py --track --capture-video
```

### Atari with Envpool

Install dependencies:
```
poetry install -E envpool
```
Train agents:
```
poetry run python ppo_atari_envpool.py
```
Train agents with experiment tracking:
```
poetry run python ppo_atari_envpool.py --track
```
Solve `Pong-v5` in 5 mins:
```
poetry run python ppo_atari_envpool.py --clip-coef=0.2 --num-envs=16 --num-minibatches=8 --num-steps=128 --update-epochs=3
```
400 game scores in `Breakout-v5` with PPO in ~1 hour (side-effects-free 3-4x speed up compared to `ppo_atari.py` with `SyncVectorEnv`):
```
poetry run python ppo_atari_envpool.py --gym-id Breakout-v5
```


### Procgen

Install dependencies:
```
poetry install -E procgen
```
Train agents:
```
poetry run python ppo_procgen.py
```
Train agents with experiment tracking:
```
poetry run python ppo_procgen.py --track
```

## Reproduction of all of our results

To reproduce the results run with `openai/baselines`, install our fork at [hhttps://github.com/vwxyzjn/baselines](hhttps://github.com/vwxyzjn/baselines). Then follow the scripts in `scripts/baselines`. To reproduce our results, follow the scripts in `scripts/ours`.


## Citation

```bibtex
@inproceedings{shengyi2022the37implementation,
  author = {Huang, Shengyi and Dossa, Rousslan Fernand Julien and Raffin, Antonin and Kanervisto, Anssi and Wang, Weixun},
  title = {The 37 Implementation Details of Proximal Policy Optimization},
  booktitle = {ICLR Blog Track},
  year = {2022},
  note = {https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/},
  url  = {https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/}
}
```
