# HalfCheetah-v2
WANDB_PROJECT=ppo-implementation-details OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run \
    --alg=ppo2 \
    --num_timesteps=1000000 \
    --env=HalfCheetah-v2 \
    --network mlp \
    --track \
    --seed 1

WANDB_PROJECT=ppo-implementation-details OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run \
    --alg=ppo2 \
    --num_timesteps=1000000 \
    --env=HalfCheetah-v2 \
    --network mlp \
    --track \
    --seed 2

WANDB_PROJECT=ppo-implementation-details OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run \
    --alg=ppo2 \
    --num_timesteps=1000000 \
    --env=HalfCheetah-v2 \
    --network mlp \
    --track \
    --seed 3

# Walker2d-v2
WANDB_PROJECT=ppo-implementation-details OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run \
    --alg=ppo2 \
    --num_timesteps=1000000 \
    --env=Walker2d-v2 \
    --network mlp \
    --track \
    --seed 1

WANDB_PROJECT=ppo-implementation-details OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run \
    --alg=ppo2 \
    --num_timesteps=1000000 \
    --env=Walker2d-v2 \
    --network mlp \
    --track \
    --seed 2

WANDB_PROJECT=ppo-implementation-details OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run \
    --alg=ppo2 \
    --num_timesteps=1000000 \
    --env=Walker2d-v2 \
    --network mlp \
    --track \
    --seed 3

# Hopper-v2
WANDB_PROJECT=ppo-implementation-details OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run \
    --alg=ppo2 \
    --num_timesteps=1000000 \
    --env=Hopper-v2 \
    --network mlp \
    --track \
    --seed 1

WANDB_PROJECT=ppo-implementation-details OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run \
    --alg=ppo2 \
    --num_timesteps=1000000 \
    --env=Hopper-v2 \
    --network mlp \
    --track \
    --seed 2

WANDB_PROJECT=ppo-implementation-details OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run \
    --alg=ppo2 \
    --num_timesteps=1000000 \
    --env=Hopper-v2 \
    --network mlp \
    --track \
    --seed 3