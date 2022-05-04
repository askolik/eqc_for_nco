import random
import time
from copy import copy

from config import BASE_PATH
from src.model.q_learning import CircuitType
from src.model.tsp_q_learning import QLearningTsp, QLearningTspAnalytical


def run_tsp(hyperparams, path):
    save = hyperparams.get('save', True)
    save_as = hyperparams.get('save_as')
    test = hyperparams.get('test', False)

    if save_as is None:
        timestamp = time.localtime()
        save_as = time.strftime("%Y-%m-%d_%H-%M-%S", timestamp) + '_' + str(random.randint(0, 1000))

    if test:
        save_as = 'dummy'

    for i in range(hyperparams.get('repetitions', 1)):
        save_as_instance = copy(save_as)
        if hyperparams.get('repetitions', 1) > 1:
            save_as_instance += f'_{i}'

        if hyperparams.get('circuit_type') == CircuitType.ANALYTIC:
            tsp_multi = QLearningTspAnalytical(
                hyperparams=hyperparams,
                save=save,
                save_as=save_as_instance,
                path=path,
                test=test)
        else:
            tsp_multi = QLearningTsp(
                hyperparams=hyperparams,
                save=save,
                save_as=save_as_instance,
                path=path,
                test=test)

        tsp_multi.perform_episodes(hyperparams.get('num_instances'))
