WANDB_PROJECT=ppo-implementation-details OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run_gym_microrts \
    --alg=ppo2 \
    --num_timesteps=2000000 \
    --env=MicrortsMining-v1 \
    --network cnn_gym_microrts \
    --num_env 8 \
    --track \
    --seed 1

WANDB_PROJECT=ppo-implementation-details OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run_gym_microrts \
    --alg=ppo2 \
    --num_timesteps=2000000 \
    --env=MicrortsMining-v1 \
    --network cnn_gym_microrts \
    --num_env 8 \
    --track \
    --seed 2

WANDB_PROJECT=ppo-implementation-details OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run_gym_microrts \
    --alg=ppo2 \
    --num_timesteps=2000000 \
    --env=MicrortsMining-v1 \
    --network cnn_gym_microrts \
    --num_env 8 \
    --track \
    --seed 3

WANDB_PROJECT=ppo-implementation-details OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run_gym_microrts \
    --alg=ppo2 \
    --num_timesteps=2000000 \
    --env=MicrortsProduceCombatUnitsShapedReward-v1 \
    --network cnn_gym_microrts \
    --num_env 8 \
    --track \
    --seed 1

WANDB_PROJECT=ppo-implementation-details OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run_gym_microrts \
    --alg=ppo2 \
    --num_timesteps=2000000 \
    --env=MicrortsProduceCombatUnitsShapedReward-v1 \
    --network cnn_gym_microrts \
    --num_env 8 \
    --track \
    --seed 2

WANDB_PROJECT=ppo-implementation-details OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run_gym_microrts \
    --alg=ppo2 \
    --num_timesteps=2000000 \
    --env=MicrortsProduceCombatUnitsShapedReward-v1 \
    --network cnn_gym_microrts \
    --num_env 8 \
    --track \
    --seed 3

WANDB_PROJECT=ppo-implementation-details OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run_gym_microrts \
    --alg=ppo2 \
    --num_timesteps=2000000 \
    --env=MicrortsAttackShapedReward-v1 \
    --network cnn_gym_microrts \
    --num_env 8 \
    --track \
    --seed 1

WANDB_PROJECT=ppo-implementation-details OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run_gym_microrts \
    --alg=ppo2 \
    --num_timesteps=2000000 \
    --env=MicrortsAttackShapedReward-v1 \
    --network cnn_gym_microrts \
    --num_env 8 \
    --track \
    --seed 2

WANDB_PROJECT=ppo-implementation-details OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run_gym_microrts \
    --alg=ppo2 \
    --num_timesteps=2000000 \
    --env=MicrortsAttackShapedReward-v1 \
    --network cnn_gym_microrts \
    --num_env 8 \
    --track \
    --seed 3

WANDB_PROJECT=ppo-implementation-details OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run_gym_microrts \
    --alg=ppo2 \
    --num_timesteps=2000000 \
    --env=MicrortsRandomEnemyShapedReward3-v1 \
    --network cnn_gym_microrts \
    --num_env 8 \
    --track \
    --seed 1

WANDB_PROJECT=ppo-implementation-details OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run_gym_microrts \
    --alg=ppo2 \
    --num_timesteps=2000000 \
    --env=MicrortsRandomEnemyShapedReward3-v1 \
    --network cnn_gym_microrts \
    --num_env 8 \
    --track \
    --seed 2

WANDB_PROJECT=ppo-implementation-details OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run_gym_microrts \
    --alg=ppo2 \
    --num_timesteps=2000000 \
    --env=MicrortsRandomEnemyShapedReward3-v1 \
    --network cnn_gym_microrts \
    --num_env 8 \
    --track \
    --seed 3
    