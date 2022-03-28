# starpilot
WANDB_PROJECT=ppo-implementation-details OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run_procgen \
    --alg=ppo2 \
    --num_timesteps=25000000 \
    --env=starpilot \
    --num_env 64 \
    --track \
    --seed 1

WANDB_PROJECT=ppo-implementation-details OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run_procgen \
    --alg=ppo2 \
    --num_timesteps=25000000 \
    --env=starpilot \
    --num_env 64 \
    --track \
    --seed 2

WANDB_PROJECT=ppo-implementation-details OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run_procgen \
    --alg=ppo2 \
    --num_timesteps=25000000 \
    --env=starpilot \
    --num_env 64 \
    --track \
    --seed 3

# bossfight
WANDB_PROJECT=ppo-implementation-details OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard  poetry run python -m baselines.run_procgen \
    --alg=ppo2 \
    --num_timesteps=25000000 \
    --env=bossfight \
    --num_env 64 \
    --track \
    --seed 1

WANDB_PROJECT=ppo-implementation-details OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard  poetry run python -m baselines.run_procgen \
    --alg=ppo2 \
    --num_timesteps=25000000 \
    --env=bossfight \
    --num_env 64 \
    --track \
    --seed 2

WANDB_PROJECT=ppo-implementation-details OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard  poetry run python -m baselines.run_procgen \
    --alg=ppo2 \
    --num_timesteps=25000000 \
    --env=bossfight \
    --num_env 64 \
    --track \
    --seed 3

# bigfish
WANDB_PROJECT=ppo-implementation-details OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run_procgen \
    --alg=ppo2 \
    --num_timesteps=25000000 \
    --env=bigfish \
    --num_env 64 \
    --track \
    --seed 1

WANDB_PROJECT=ppo-implementation-details OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run_procgen \
    --alg=ppo2 \
    --num_timesteps=25000000 \
    --env=bigfish \
    --num_env 64 \
    --track \
    --seed 2

WANDB_PROJECT=ppo-implementation-details OPENAI_LOGDIR=$PWD/runs OPENAI_LOG_FORMAT=tensorboard poetry run python -m baselines.run_procgen \
    --alg=ppo2 \
    --num_timesteps=25000000 \
    --env=bigfish \
    --num_env 64 \
    --track \
    --seed 3