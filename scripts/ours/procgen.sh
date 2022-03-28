# starpilot
poetry run python ppg_procgen.py --gym-id starpilot --track --seed 1 --wandb-project-name ppo-implementation-details
poetry run python ppo_procgen.py --gym-id starpilot --track --seed 2 --wandb-project-name ppo-implementation-details
poetry run python ppo_procgen.py --gym-id starpilot --track --seed 3 --wandb-project-name ppo-implementation-details

# bossfight
poetry run python ppo_procgen.py --gym-id bossfight --track --seed 1 --wandb-project-name ppo-implementation-details
poetry run python ppo_procgen.py --gym-id bossfight --track --seed 2 --wandb-project-name ppo-implementation-details
poetry run python ppo_procgen.py --gym-id bossfight --track --seed 3 --wandb-project-name ppo-implementation-details

# bigfish
poetry run python ppo_procgen.py --gym-id bigfish --track --seed 1 --wandb-project-name ppo-implementation-details
poetry run python ppo_procgen.py --gym-id bigfish --track --seed 2 --wandb-project-name ppo-implementation-details
poetry run python ppo_procgen.py --gym-id bigfish --track --seed 3 --wandb-project-name ppo-implementation-details 
