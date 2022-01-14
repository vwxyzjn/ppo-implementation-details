# BreakoutNoFrameskip-v4
WANDB_PROJECT=ppo-implementation-details OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run \
    --alg=ppo2 \
    --num_timesteps=10000000 \
    --env=BreakoutNoFrameskip-v4 \
    --network cnn \
    --num_env 8 \
    --track \
    --seed 1

WANDB_PROJECT=ppo-implementation-details OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run \
    --alg=ppo2 \
    --num_timesteps=10000000 \
    --env=BreakoutNoFrameskip-v4 \
    --network cnn \
    --num_env 8 \
    --track \
    --seed 2

WANDB_PROJECT=ppo-implementation-details OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run \
    --alg=ppo2 \
    --num_timesteps=10000000 \
    --env=BreakoutNoFrameskip-v4 \
    --network cnn \
    --num_env 8 \
    --track \
    --seed 3

# PongNoFrameskip-v4
WANDB_PROJECT=ppo-implementation-details OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard  poetry run python -m baselines.run \
    --alg=ppo2 \
    --num_timesteps=10000000 \
    --env=PongNoFrameskip-v4 \
    --network cnn \
    --num_env 8 \
    --track \
    --seed 1

WANDB_PROJECT=ppo-implementation-details OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard  poetry run python -m baselines.run \
    --alg=ppo2 \
    --num_timesteps=10000000 \
    --env=PongNoFrameskip-v4 \
    --network cnn \
    --num_env 8 \
    --track \
    --seed 2

WANDB_PROJECT=ppo-implementation-details OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard  poetry run python -m baselines.run \
    --alg=ppo2 \
    --num_timesteps=10000000 \
    --env=PongNoFrameskip-v4 \
    --network cnn \
    --num_env 8 \
    --track \
    --seed 3

# BeamRiderNoFrameskip-v4
WANDB_PROJECT=ppo-implementation-details OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run \
    --alg=ppo2 \
    --num_timesteps=10000000 \
    --env=BeamRiderNoFrameskip-v4 \
    --network cnn \
    --num_env 8 \
    --track \
    --seed 1

WANDB_PROJECT=ppo-implementation-details OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run \
    --alg=ppo2 \
    --num_timesteps=10000000 \
    --env=BeamRiderNoFrameskip-v4 \
    --network cnn \
    --num_env 8 \
    --track \
    --seed 2

WANDB_PROJECT=ppo-implementation-details OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run \
    --alg=ppo2 \
    --num_timesteps=10000000 \
    --env=BeamRiderNoFrameskip-v4 \
    --network cnn \
    --num_env 8 \
    --track \
    --seed 3