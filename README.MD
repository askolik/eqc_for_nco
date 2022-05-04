# Equivariant quantum circuits for neural combinatorial optimization

This repository contains the code that was used to generate the numerical results in arXiv_link.

In this work, we investigate the advantages gained by a symmetry-preserving ansatz in a quantum reinforcement learning setting. To run a circiut type of your choice, execute one of the _run_circuit_type.py_ files after installing all required packages in _requirements.txt_.

E.g., to execute the EQC run _run_eqc.py_ and adjust the following hyperparameters of your choice:

    hyperparams = {
        'n_vars': 10,  # number of cities
        'episodes': 5000,  # number of episodes
        'batch_size': 10,  # number of training samples in batch
        'epsilon': 1,  # initial value for epsilon-greedy exploration
        'epsilon_decay': 0.99,  # decay factor for epsilon in each episode
        'epsilon_min': 0.01,  # no epsilon decay below this value
        'gamma': 0.9,  # discount factor
        'update_after': 10,  # update frequency of main model
        'update_target_after': 30,  # update frequency of target model
        'learning_rate_in': 0.00001,  # learning rate of Adam optimizer
        'n_layers': 1,  # number of EQC layers
        'num_instances': 100,  # number of training instances to use
        'circuit_type': CircuitType.EQC,
        'data_path': BASE_PATH + 'tsp/tsp_10_train/tsp_10_reduced_train.pickle',
        'repetitions': 1,  # how often to execute this configuratoin
        'save': False,  # save data about training performance and model
        'test': True  # True for more verbose output during training
    }  