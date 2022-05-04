import tensorflow as tf

from src.utils.limit_thread_usage import set_thread_usage_limit
set_thread_usage_limit(10, tf)

from src.model.q_learning import CircuitType
from run import run_tsp
from config import BASE_PATH


hyperparams = {
    'n_vars': 10,
    'episodes': 5000,
    'batch_size': 10,
    'epsilon': 1,
    'epsilon_decay': 0.99,
    'epsilon_min': 0.01,
    'gamma': 0.9,
    'update_after': 10,
    'update_target_after': 30,
    'learning_rate': 0.00001,
    'epsilon_schedule': 'fast',
    'memory_length': 10000,
    'num_instances': 100,
    'circuit_type': CircuitType.ANALYTIC,
    'data_path': BASE_PATH + 'tsp/tsp_10_train/tsp_10_reduced_train.pickle',
    'repetitions': 1,
    'save': False,
    'test': True
}


if __name__ == '__main__':
    run_tsp(hyperparams, '/save_path/')
