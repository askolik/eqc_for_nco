import copy
import pickle
import random
from itertools import combinations
from random import choice, choices

import cirq
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq
import numpy as np

from config import BASE_PATH
from src.utils.analytic_exp_vals import compute_analytic_expectation
from src.utils.helpers import compute_tour_length
from src.model.circuits import graph_encoding_circuit, hardware_efficient_circuit
from src.model.layers import TrainableRescaling, ScalableDataReuploadingController, EquivariantLayer
from src.model.q_learning import QLearning, CircuitType
from collections import namedtuple


class QLearningTsp(QLearning):
    def __init__(
            self,
            hyperparams,
            save=True,
            save_as=None,
            test=False,
            path=BASE_PATH):

        super(QLearningTsp, self).__init__(hyperparams, save, save_as, test, path)

        self.fully_connected_qubits = list(combinations(list(range(self.n_vars)), 2))
        self.qubits = cirq.GridQubit.rect(1, self.n_vars)
        self.is_multi_instance = hyperparams.get('is_multi_instance')
        self.readout_op = self.initialize_readout()
        self.interaction = namedtuple(
            'interaction', ('state', 'action', 'reward', 'next_state', 'done', 'partial_tour', 'edge_weights'))
        self.model, self.target_model = self.initialize_models()
        self.data_path = hyperparams.get('data_path')

    def save_data(self, meta, tour_lengths, optimal_tour_lengths):
        with open(self.path + '{}_meta.pickle'.format(self.save_as), 'wb') as file:
            pickle.dump(meta, file)

        with open(self.path + '{}_tour_lengths.pickle'.format(self.save_as), 'wb') as file:
            pickle.dump(tour_lengths, file)

        with open(self.path + '{}_optimal_lengths.pickle'.format(self.save_as), 'wb') as file:
            pickle.dump(optimal_tour_lengths, file)

        self.model.save_weights(self.path + '{}_model.h5'.format(self.save_as))

    def initialize_readout(self):
        readout_ops = []
        for edge in self.fully_connected_qubits:
            readout_ops.append(cirq.Z(self.qubits[edge[0]]) * cirq.Z(self.qubits[edge[1]]))
        return readout_ops

    def generate_neqc_model(self, is_target_model=False):
        name_prefix = ''
        if is_target_model:
            name_prefix = 'target_'

        num_edges_in_graph = len(self.fully_connected_qubits)
        n_input_params = self.n_vars + num_edges_in_graph
        n_var_params = 15 * (self.n_vars - 1) * self.n_pred_layers

        n_data_reps = self.n_layers if self.use_reuploading else 1

        data_symbols = [
            [
                sympy.Symbol(f'd_{layer}_{qubit}')
                for qubit in range(len(self.qubits))] +
            [[sympy.Symbol(f'd_{layer}_e_{ew}') for ew in range(num_edges_in_graph)]]
            for layer in range(n_data_reps)]

        circuit = graph_encoding_circuit(
            self.fully_connected_qubits, self.qubits, self.n_layers, data_symbols)

        input_data = tf.keras.Input(shape=n_input_params, dtype=tf.dtypes.float32, name='input')
        input_q_state = tf.keras.Input(shape=(), dtype=tf.string, name='quantum_state')

        flattened_data_symbols = []
        for layer in data_symbols:
            for item in layer:
                if type(item) == list:
                    for symbol in item:
                        flattened_data_symbols.append(str(symbol))
                else:
                    flattened_data_symbols.append(str(item))

        encoding_layer = ScalableDataReuploadingController(
            num_input_params=n_input_params, num_params=n_var_params, circuit_depth=self.n_layers,
            params=flattened_data_symbols, trainable_scaling=self.trainable_scaling,
            use_reuploading=self.use_reuploading)

        expectation_layer = tfq.layers.ControlledPQC(
            circuit, differentiator=tfq.differentiators.Adjoint(),
            operators=self.readout_op, name="PQC")

        expectation_values = expectation_layer(
            [input_q_state, encoding_layer(input_data)])

        output_extension_layer = tf.keras.Sequential([
            TrainableRescaling(
                len(self.readout_op),
                trainable_obs_weight=self.trainable_obs_weight)
        ])

        output = output_extension_layer(expectation_values)

        model = tf.keras.Model(
            inputs=[input_q_state, input_data],
            outputs=output, name=f'{name_prefix}q_model')

        model.compile(
            optimizer=self.optimizer,
            loss=tf.keras.losses.mse)

        if self.test:
            model.summary()

        return model

    def generate_eqc_model(self, is_target_model=False):
        name_prefix = ''
        if is_target_model:
            name_prefix = 'target_'

        num_edges_in_graph = len(self.fully_connected_qubits)
        n_input_params = self.n_vars + num_edges_in_graph

        data_symbols = [
            [
                sympy.Symbol(f'd_{layer}_{qubit}')
                for qubit in range(len(self.qubits))] +
            [[sympy.Symbol(f'd_{layer}_e_{ew}') for ew in range(num_edges_in_graph)]]
            for layer in range(self.n_layers)]

        circuit = graph_encoding_circuit(
            self.fully_connected_qubits, self.qubits, self.n_layers, data_symbols)

        input_data = tf.keras.Input(shape=n_input_params, dtype=tf.dtypes.float32, name='input')
        input_q_state = tf.keras.Input(shape=(), dtype=tf.string, name='quantum_state')

        flattened_data_symbols = []
        for layer in data_symbols:
            for item in layer:
                if type(item) == list:
                    for symbol in item:
                        flattened_data_symbols.append(str(symbol))
                else:
                    flattened_data_symbols.append(str(item))

        encoding_layer = EquivariantLayer(
            num_input_params=n_input_params, n_vars=self.n_vars,
            n_edges=num_edges_in_graph, circuit_depth=self.n_layers,
            params=flattened_data_symbols)

        expectation_layer = tfq.layers.ControlledPQC(
            circuit, differentiator=tfq.differentiators.Adjoint(),
            operators=self.readout_op, name="PQC")

        expectation_values = expectation_layer(
            [input_q_state, encoding_layer(input_data)])

        output_extension_layer = tf.keras.Sequential([
            TrainableRescaling(
                len(self.readout_op),
                trainable_obs_weight=self.trainable_obs_weight)
        ])

        output = output_extension_layer(expectation_values)

        model = tf.keras.Model(
            inputs=[input_q_state, input_data],
            outputs=output, name=f'{name_prefix}q_model')

        model.compile(
            optimizer=self.optimizer,
            loss=tf.keras.losses.mse)

        if self.test:
            model.summary()

        return model

    def generate_hwe_model(self, is_target_model=False):
        name_prefix = ''
        if is_target_model:
            name_prefix = 'target_'

        num_edges_in_graph = len(self.fully_connected_qubits)
        n_input_params = self.n_vars + num_edges_in_graph
        n_var_params = self.n_vars * self.n_layers
        n_data_reps = self.n_layers if self.use_reuploading else 1

        symbols = [
            [
                sympy.Symbol(f'theta_{layer}_{param}')
                for param in range(len(self.qubits)*1)]
            for layer in range(self.n_layers)]

        data_symbols = [
            [
                sympy.Symbol(f'd_{layer}_{qubit}')
                for qubit in range(len(self.qubits))] +
            [[sympy.Symbol(f'd_{layer}_e_{ew}') for ew in range(num_edges_in_graph)]]
            for layer in range(n_data_reps)]

        circuit = hardware_efficient_circuit(
            self.fully_connected_qubits, self.qubits, self.n_layers,
            symbols, data_symbols, use_reuploading=self.use_reuploading)

        input_data = tf.keras.Input(shape=n_input_params, dtype=tf.dtypes.float32, name='input')
        input_q_state = tf.keras.Input(shape=(), dtype=tf.string, name='quantum_state')

        flattened_data_symbols = []
        for layer in data_symbols:
            for item in layer:
                if type(item) == list:
                    for symbol in item:
                        flattened_data_symbols.append(str(symbol))
                else:
                    flattened_data_symbols.append(str(item))

        flattened_symbols = []
        for layer_item in symbols:
            for item in layer_item:
                if type(item) == list:
                    for symbol in item:
                        flattened_symbols.append(str(symbol))
                else:
                    flattened_symbols.append(str(item))

        encoding_layer = ScalableDataReuploadingController(
            num_input_params=n_input_params, num_params=n_var_params, circuit_depth=self.n_layers,
            params=flattened_data_symbols + flattened_symbols,
            trainable_scaling=self.trainable_scaling, use_reuploading=self.use_reuploading)

        expectation_layer = tfq.layers.ControlledPQC(
            circuit, differentiator=tfq.differentiators.Adjoint(),
            operators=self.readout_op, name="PQC")

        expectation_values = expectation_layer(
            [input_q_state, encoding_layer(input_data)])

        output_extension_layer = tf.keras.Sequential([
            TrainableRescaling(
                len(self.readout_op),
                trainable_obs_weight=self.trainable_obs_weight)
        ])

        output = output_extension_layer(expectation_values)

        model = tf.keras.Model(
            inputs=[input_q_state, input_data],
            outputs=output, name=f'{name_prefix}q_model')

        model.compile(
            optimizer=self.optimizer,
            loss=tf.keras.losses.mse)

        if self.test:
            model.summary()

        return model

    def initialize_models(self):
        model = target_model = None
        if self.circuit_type == CircuitType.NEQC:
            model = self.generate_neqc_model()
            target_model = self.generate_neqc_model(is_target_model=True)
            target_model.set_weights(model.get_weights())
        elif self.circuit_type == CircuitType.EQC:
            model = self.generate_eqc_model()
            target_model = self.generate_eqc_model(is_target_model=True)
            target_model.set_weights(model.get_weights())
        elif self.circuit_type == CircuitType.HWE:
            model = self.generate_hwe_model()
            target_model = self.generate_hwe_model(is_target_model=True)
            target_model.set_weights(model.get_weights())

        if self.test:
            model.summary()

        return model, target_model

    @staticmethod
    def graph_to_list(
            nodes, fully_connected_edges, edge_weights, available_nodes, node_to_qubit_map):
        vals = []
        for node in nodes:
            vals.append(int(node_to_qubit_map[node] in available_nodes) * np.pi)

        for edge in fully_connected_edges:
            vals.append(np.arctan(edge_weights[edge]))

        return vals

    def q_vals_from_expectations(self, partial_tours, edge_weights, expectations):
        expectations = expectations.numpy()
        indexed_expectations = []
        for exps in expectations:
            batch_ix_exp = {}
            for edge, exp_val in zip(self.fully_connected_qubits, exps):
                batch_ix_exp[edge] = exp_val
            indexed_expectations.append(batch_ix_exp)

        batch_q_vals = []
        for tour_ix, partial_tour in enumerate(partial_tours):
            q_vals = []
            for i in range(self.n_vars):
                node_in_tour = False
                for edge in partial_tour:
                    if i in edge:
                        node_in_tour = True

                if not node_in_tour:
                    next_edge = None
                    if partial_tour:
                        next_edge = (partial_tour[-1][1], i)
                    else:
                        if i > 0:
                            next_edge = (0, i)

                    if next_edge is not None:
                        try:
                            q_val = edge_weights[tour_ix][next_edge] * indexed_expectations[tour_ix][next_edge]
                        except KeyError:
                            q_val = edge_weights[tour_ix][
                                (next_edge[1], next_edge[0])] * indexed_expectations[tour_ix][
                                        (next_edge[1], next_edge[0])]
                    else:
                        q_val = -10000
                else:
                    q_val = -10000
                q_vals.append(q_val)

            batch_q_vals.append(q_vals)

        return np.asarray(batch_q_vals)

    def get_action(self, state_tensor, available_nodes, partial_tour, edge_weights):
        if np.random.uniform() < self.epsilon:
            action = choice(available_nodes)
        else:
            state_tensor = tf.convert_to_tensor(state_tensor)
            state_tensor = tf.expand_dims(state_tensor, 0)
            expectations = self.model([tfq.convert_to_tensor([cirq.Circuit()]), state_tensor])
            q_vals = self.q_vals_from_expectations([partial_tour], [edge_weights], expectations)[0]
            action = np.argmax(q_vals)

        return action

    @staticmethod
    def get_masks_for_actions(edge_weights, partial_tours):
        batch_masks = []
        for tour_ix, partial_tour in enumerate(partial_tours):
            mask = []
            for edge, weight in edge_weights[tour_ix].items():
                if edge in partial_tour or (edge[1], edge[0]) in partial_tour:
                    mask.append(weight)
                else:
                    mask.append(0)

            batch_masks.append(mask)

        return np.asarray(batch_masks)

    @staticmethod
    def cost(nodes, tour):
        return -compute_tour_length(nodes, tour)

    def compute_reward(self, nodes, old_state, state):
        return self.cost(nodes, state) - self.cost(nodes, old_state)

    def train_step(self):
        training_batch = choices(self.memory, k=self.batch_size)
        training_batch = self.interaction(*zip(*training_batch))

        states = [x for x in training_batch.state]
        rewards = np.asarray([x for x in training_batch.reward], dtype=np.float32)
        next_states = [x for x in training_batch.next_state]
        done = np.asarray([x for x in training_batch.done])
        partial_tours = [x for x in training_batch.partial_tour]
        edge_weights = [x for x in training_batch.edge_weights]

        states = tf.convert_to_tensor(states)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float64)
        next_states = tf.convert_to_tensor(next_states)
        done = tf.convert_to_tensor(done, dtype=tf.float64)

        exp_values_future = self.model([tfq.convert_to_tensor([cirq.Circuit()] * self.batch_size), next_states])
        future_rewards = tf.convert_to_tensor(self.q_vals_from_expectations(
            partial_tours, edge_weights, exp_values_future), dtype=tf.float64)

        target_q_values = rewards + (
                self.gamma * tf.reduce_max(future_rewards, axis=1) * (1.0 - done))

        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            exp_values = self.model([tfq.convert_to_tensor([cirq.Circuit()]*self.batch_size), states])
            exp_val_masks = self.get_masks_for_actions(edge_weights, partial_tours)
            q_values_masked = tf.reduce_sum(tf.multiply(exp_values, exp_val_masks), axis=1)
            loss = self.loss_fun(target_q_values, q_values_masked)

        grads = tape.gradient(loss, self.model.trainable_variables)

        if len(self.optimizers) == 1:
            self.optimizers[0].apply_gradients(zip(grads, self.model.trainable_weights))
        else:
            for optimizer, w in zip(self.optimizers, self.w_idx):
                optimizer.apply_gradients([(grads[w], self.model.trainable_variables[w])])

        return loss.numpy()

    def perform_episodes(self, num_instances):
        self.meta['num_instances'] = num_instances
        self.meta['best_tour_length'] = 100000
        self.meta['best_tour'] = []
        self.meta['best_tour_ix'] = 0
        self.meta['env_solved'] = False

        with open(self.data_path, 'rb') as file:
            data = pickle.load(file)

        if self.is_multi_instance:
            x_train = data['x_train'][:num_instances]
            y_train = data['y_train'][:num_instances]
        else:
            x_train = data['x_train']
            y_train = data['y_train']

        tour_length_history = []
        optimal_tour_length_history = []
        ratio_history = []
        running_avgs = []
        running_avg = 0

        for episode in range(self.episodes):
            instance_number = random.randint(0, num_instances-1)
            tsp_graph_nodes = x_train[instance_number]
            optimal_tour_length = compute_tour_length(
                tsp_graph_nodes, [int(x - 1) for x in y_train[instance_number][:-1]])
            node_to_qubit_map = {}
            for i, node in enumerate(tsp_graph_nodes):
                node_to_qubit_map[node] = i

            fully_connected_edges = []
            edge_weights = {}
            edge_weights_ix = {}
            for edge in self.fully_connected_qubits:
                fully_connected_edges.append((tsp_graph_nodes[edge[0]], tsp_graph_nodes[edge[1]]))
                edge_distance = np.linalg.norm(
                    np.asarray(tsp_graph_nodes[edge[0]]) - np.asarray(tsp_graph_nodes[edge[1]]))
                edge_weights[(tsp_graph_nodes[edge[0]], tsp_graph_nodes[edge[1]])] = edge_distance
                edge_weights_ix[edge] = edge_distance

            tour = [0]  # w.l.o.g. we always start at city 0
            tour_edges = []
            step_rewards = []
            available_nodes = list(range(1, self.n_vars))

            for i in range(self.n_vars):
                prev_tour = copy.deepcopy(tour)
                state_list = self.graph_to_list(
                    tsp_graph_nodes, fully_connected_edges, edge_weights,
                    available_nodes, node_to_qubit_map)

                next_node = self.get_action(state_list, available_nodes, tour_edges, edge_weights_ix)
                tour_edges.append((tour[-1], next_node))
                new_tour_edges = copy.deepcopy(tour_edges)
                tour.append(next_node)

                remove_node_ix = available_nodes.index(next_node)
                del available_nodes[remove_node_ix]

                if len(tour) > 1:
                    reward = self.compute_reward(tsp_graph_nodes, prev_tour, tour)
                    step_rewards.append(reward)

                    done = 0 if len(available_nodes) > 1 else 1
                    transition = (state_list, next_node, reward, self.graph_to_list(
                        tsp_graph_nodes, fully_connected_edges, edge_weights,
                        available_nodes, node_to_qubit_map), done, new_tour_edges, edge_weights_ix)
                    self.memory.append(transition)

                if len(available_nodes) == 1:
                    prev_tour = copy.deepcopy(tour)
                    tour_edges.append((tour[-1], available_nodes[0]))
                    tour_edges.append((available_nodes[0], tour[0]))
                    new_tour_edges = copy.deepcopy(tour_edges)
                    tour.append(available_nodes[0])
                    tour.append(tour[0])
                    reward = self.compute_reward(tsp_graph_nodes, prev_tour, tour)
                    step_rewards.append(reward)

                    transition = (state_list, next_node, reward, self.graph_to_list(
                        tsp_graph_nodes, fully_connected_edges, edge_weights,
                        available_nodes, node_to_qubit_map), 1, new_tour_edges, edge_weights_ix)
                    self.memory.append(transition)
                    break

            tour_length = compute_tour_length(tsp_graph_nodes, tour)
            tour_length_history.append(tour_length)
            optimal_tour_length_history.append(optimal_tour_length)

            if tour_length < self.meta.get('best_tour_length'):
                self.meta['best_tour_length'] = tour_length
                self.meta['best_tour'] = tour
                self.meta['best_tour_ix'] = instance_number

            if len(self.memory) >= self.batch_size:
                if episode % self.update_after == 0:
                    loss = self.train_step()
                    print(
                        f"Episode {episode}, loss {loss}, running avg {running_avg}, epsilon {self.epsilon}")
                    print(f"\tFinal tour: {tour}")
                else:
                    print(
                            f"Episode {episode}, running avg {running_avg}, epsilon {self.epsilon}")
                    print(f"\tFinal tour: {tour}")

                if episode % self.update_target_after == 0:
                    self.target_model.set_weights(self.model.get_weights())

            if self.epsilon_schedule == 'fast':
                self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

            if self.save:
                self.save_data(self.meta, tour_length_history, optimal_tour_length_history)

            ratio_history.append(tour_length_history[-1] / optimal_tour_length)

            if len(ratio_history) >= 100:
                running_avg = np.mean(ratio_history[-100:])
            else:
                running_avg = np.mean(ratio_history)

            running_avgs.append(running_avg)

            if len(ratio_history) >= 100 and running_avg <= 1.05:
                print(f"Environment solved in {episode+1} episodes!")
                self.meta['env_solved'] = True
                if self.save:
                    self.save_data(self.meta, tour_length_history, optimal_tour_length_history)
                break

        if self.test:
            import matplotlib.pyplot as plt
            plt.plot(running_avgs)
            plt.ylabel("Ratio to optimal tour length")
            plt.xlabel("Episode")
            plt.title("Running average over past 100 episodes")
            plt.show()


class QLearningTspAnalytical(QLearning):
    def __init__(
            self,
            hyperparams,
            save=True,
            save_as=None,
            test=False,
            path=BASE_PATH):

        super(QLearningTspAnalytical, self).__init__(hyperparams, save, save_as, test, path)

        self.interaction = namedtuple(
            'interaction', ('state', 'action', 'reward', 'next_state', 'done', 'partial_tour', 'edge_weights'))
        self.data_path = hyperparams.get('data_path')

        self.epsilon_fd = 0.005  # epsilon for finite difference gradient

        # Adam hyperparams
        self.t = {}
        self.m_t = {}
        self.v_t = {}
        self.beta_1 = 0.9
        self.beta_2 = 0.999

        self.params = [1.1, 1]
        self.target_params = [1.1, 1]
        self.fully_connected_edges = list(combinations(list(range(self.n_vars)), 2))

    def save_data(self, meta, tour_lengths, optimal_tour_lengths):
        with open(self.path + '{}_meta.pickle'.format(self.save_as), 'wb') as file:
            pickle.dump(meta, file)

        with open(self.path + '{}_tour_lengths.pickle'.format(self.save_as), 'wb') as file:
            pickle.dump(tour_lengths, file)

        with open(self.path + '{}_optimal_lengths.pickle'.format(self.save_as), 'wb') as file:
            pickle.dump(optimal_tour_lengths, file)

        with open(self.path + '{}_params.pickle'.format(self.save_as), 'wb') as file:
            pickle.dump(self.params, file)

    def q_vals_from_expectations(self, partial_tours, edge_weights, params):
        batch_q_vals = []
        for tour_ix, partial_tour in enumerate(partial_tours):
            q_vals = []
            for i in range(self.n_vars):
                node_in_tour = False
                for edge in partial_tour:
                    if i in edge:
                        node_in_tour = True

                if not node_in_tour:
                    next_edge = None
                    if partial_tour:
                        next_edge = (partial_tour[-1][1], i)
                    else:
                        if i > 0:
                            next_edge = (0, i)

                    if next_edge is not None:
                        try:
                            weight = edge_weights[tour_ix][next_edge]
                        except KeyError:
                            weight = edge_weights[tour_ix][
                                (next_edge[1], next_edge[0])]
                        q_val = weight * compute_analytic_expectation(
                            params[0], params[1], next_edge[0], next_edge[1],
                            self.n_vars, edge_weights[tour_ix], available_node='j')
                    else:
                        q_val = -10000
                else:
                    q_val = -10000
                q_vals.append(q_val)

            batch_q_vals.append(q_vals)

        return np.asarray(batch_q_vals)

    def get_action(self, available_nodes, partial_tour, edge_weights):
        if np.random.uniform() < self.epsilon:
            action = choice(available_nodes)
        else:
            q_vals = self.q_vals_from_expectations(
                [partial_tour], [edge_weights], self.params)[0]
            action = np.argmax(q_vals)

        return action

    @staticmethod
    def cost(nodes, tour):
        return -compute_tour_length(nodes, tour)

    def compute_reward(self, nodes, old_state, state):
        return self.cost(nodes, state) - self.cost(nodes, old_state)

    def compute_fd_loss_gradient(self, states):
        batch_gradients = {0: [], 1: []}
        loss_vals = []
        for partial_tour, edge_weights, action, target_q_value in states:
            q_vals = self.q_vals_from_expectations([partial_tour], [edge_weights], self.params)[0]
            loss = abs(target_q_value - [q_vals[action]])**2
            loss_vals.append(loss)
            for i, params in enumerate(self.params):
                shifted_params = copy.deepcopy(self.params)
                shifted_params[i] += self.epsilon_fd
                q_vals_shifted = self.q_vals_from_expectations([partial_tour], [edge_weights], shifted_params)[0]
                derivative = (abs(target_q_value - [q_vals_shifted[action]]) - loss) / self.epsilon_fd
                batch_gradients[i].append(derivative.numpy()[0])

        gradient = {0: np.mean(batch_gradients[0]), 1: np.mean(batch_gradients[1])}
        return gradient, np.mean(loss_vals)

    def train_step(self):
        training_batch = choices(self.memory, k=self.batch_size)
        training_batch = self.interaction(*zip(*training_batch))

        actions = [x for x in training_batch.action]
        rewards = np.asarray([x for x in training_batch.reward], dtype=np.float32)
        done = np.asarray([x for x in training_batch.done])
        partial_tours = [x for x in training_batch.partial_tour]
        edge_weights = [x for x in training_batch.edge_weights]

        future_q_vals_all = self.q_vals_from_expectations(
            partial_tours, edge_weights, self.target_params)
        future_q_vals = []
        for i, q_vals in enumerate(future_q_vals_all):
            future_q_vals.append(q_vals[actions[i]])
        future_q_vals = np.asarray(future_q_vals)
        target_q_values = rewards + (
                self.gamma * tf.reduce_max(future_q_vals, axis=0) * (1.0 - done))

        gradient, loss = self.compute_fd_loss_gradient(
            zip(partial_tours, edge_weights, actions, target_q_values))
        updated_params = copy.deepcopy(self.params)
        for param, deriv in gradient.items():
            self.t[param] = self.t.get(param, 0) + 1
            self.m_t[param] = (self.beta_1 * self.m_t.get(param, 0)) + ((1 - self.beta_1) * deriv)
            self.v_t[param] = (self.beta_2 * self.v_t.get(param, 0)) + ((1 - self.beta_2) * deriv ** 2)
            m_cap = self.m_t[param] / (1 - self.beta_1 ** self.t[param])
            v_cap = self.v_t[param] / (1 - self.beta_2 ** self.t[param])
            v_temp = np.sqrt(v_cap) + self.epsilon
            grad = 1 * m_cap / v_temp
            parameter_update = -1 * self.learning_rate * grad
            updated_params[param] = updated_params[param] + parameter_update

        self.params = updated_params

        return loss

    def perform_episodes(self, num_instances):
        self.meta['num_instances'] = num_instances
        self.meta['best_tour_length'] = 100000
        self.meta['best_tour'] = []
        self.meta['best_tour_ix'] = 0
        self.meta['env_solved'] = False

        with open(self.data_path, 'rb') as file:
            data = pickle.load(file)

        x_train = data['x_train'][:num_instances]
        y_train = data['y_train'][:num_instances]

        tour_length_history = []
        optimal_tour_length_history = []
        ratio_history = []
        running_avgs = []
        running_avg = 0

        for episode in range(self.episodes):
            instance_number = random.randint(0, num_instances-1)
            tsp_graph_nodes = x_train[instance_number]
            optimal_tour_length = compute_tour_length(
                tsp_graph_nodes, [int(x - 1) for x in y_train[instance_number][:-1]])
            node_to_qubit_map = {}
            for i, node in enumerate(tsp_graph_nodes):
                node_to_qubit_map[node] = i

            fully_connected_edges = []
            edge_weights_ix = {}
            for edge in self.fully_connected_edges:
                fully_connected_edges.append((tsp_graph_nodes[edge[0]], tsp_graph_nodes[edge[1]]))
                edge_distance = np.linalg.norm(
                    np.asarray(tsp_graph_nodes[edge[0]]) - np.asarray(tsp_graph_nodes[edge[1]]))
                edge_weights_ix[edge] = edge_distance

            tour = [0]  # w.l.o.g. we always start at city 0
            tour_edges = []
            step_rewards = []
            available_nodes = list(range(1, self.n_vars))

            for i in range(self.n_vars):
                prev_tour = copy.deepcopy(tour)
                next_node = self.get_action(available_nodes, tour_edges, edge_weights_ix)
                tour_edges.append((tour[-1], next_node))
                new_tour_edges = copy.deepcopy(tour_edges)
                tour.append(next_node)

                remove_node_ix = available_nodes.index(next_node)
                del available_nodes[remove_node_ix]

                if len(tour) > 1:
                    reward = self.compute_reward(tsp_graph_nodes, prev_tour, tour)
                    step_rewards.append(reward)

                    done = 0 if len(available_nodes) > 1 else 1
                    transition = (
                        edge_weights_ix, next_node, reward, edge_weights_ix,
                        done, new_tour_edges, edge_weights_ix)
                    self.memory.append(transition)

                if len(available_nodes) == 1:
                    prev_tour = copy.deepcopy(tour)
                    tour_edges.append((tour[-1], available_nodes[0]))
                    tour_edges.append((available_nodes[0], tour[0]))
                    new_tour_edges = copy.deepcopy(tour_edges)
                    tour.append(available_nodes[0])
                    tour.append(tour[0])
                    reward = self.compute_reward(tsp_graph_nodes, prev_tour, tour)
                    step_rewards.append(reward)

                    transition = (
                        edge_weights_ix, next_node, reward, edge_weights_ix, 1,
                        new_tour_edges, edge_weights_ix)
                    self.memory.append(transition)
                    break

            tour_length = compute_tour_length(tsp_graph_nodes, tour)
            tour_length_history.append(tour_length)
            optimal_tour_length_history.append(optimal_tour_length)

            if tour_length < self.meta.get('best_tour_length'):
                self.meta['best_tour_length'] = tour_length
                self.meta['best_tour'] = tour
                self.meta['best_tour_ix'] = instance_number

            if len(self.memory) >= self.batch_size:
                if episode % self.update_after == 0:
                    loss = self.train_step()
                    print(
                        f"Episode {episode}, loss {loss}, running avg {running_avg}, epsilon {self.epsilon}")
                    print(f"\tFinal tour: {tour}")
                else:
                    print(
                            f"Episode {episode}, running avg {running_avg}, epsilon {self.epsilon}")
                    print(f"\tFinal tour: {tour}")

                if episode % self.update_target_after == 0:
                    self.target_params = copy.deepcopy(self.params)

            self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

            if self.save:
                self.save_data(self.meta, tour_length_history, optimal_tour_length_history)

            ratio_history.append(tour_length_history[-1] / optimal_tour_length)

            if len(ratio_history) >= 100:
                running_avg = np.mean(ratio_history[-100:])
            else:
                running_avg = np.mean(ratio_history)

            running_avgs.append(running_avg)

            if len(ratio_history) >= 100 and running_avg <= 1.05:
                print(f"Environment solved in {episode+1} episodes!")
                self.meta['env_solved'] = True
                if self.save:
                    self.save_data(self.meta, tour_length_history, optimal_tour_length_history)
                break

        if self.test:
            import matplotlib.pyplot as plt
            plt.plot(running_avgs)
            plt.ylabel("Ratio to optimal tour length")
            plt.xlabel("Episode")
            plt.title("Running average over past 100 episodes")
            plt.show()


def get_tour_from_trained_model(path, tsp_q, tsp_graph_nodes):
    model = tsp_q.model
    model.load_weights(path)

    tour = [0]
    tour_edges = []
    available_nodes = list(range(1, len(tsp_graph_nodes)))

    node_to_qubit_map = {}
    for i, node in enumerate(tsp_graph_nodes):
        node_to_qubit_map[node] = i

    fully_connected_edges = []
    fully_connected_edges_ix = list(combinations(list(range(len(tsp_graph_nodes))), 2))
    edge_weights = {}
    edge_weights_ix = {}
    for edge in fully_connected_edges_ix:
        fully_connected_edges.append((tsp_graph_nodes[edge[0]], tsp_graph_nodes[edge[1]]))
        edge_distance = np.linalg.norm(np.asarray(tsp_graph_nodes[edge[0]]) - np.asarray(tsp_graph_nodes[edge[1]]))
        edge_weights[(tsp_graph_nodes[edge[0]], tsp_graph_nodes[edge[1]])] = edge_distance
        edge_weights_ix[edge] = edge_distance

    for i in range(len(tsp_graph_nodes)):
        state_list = tsp_q.graph_to_list(
            tsp_graph_nodes, fully_connected_edges, edge_weights,
            available_nodes, node_to_qubit_map)

        next_node = tsp_q.get_action(state_list, available_nodes, tour_edges, edge_weights_ix)
        tour_edges.append((tour[-1], next_node))
        tour.append(next_node)

        remove_node_ix = available_nodes.index(next_node)
        del available_nodes[remove_node_ix]

        if len(available_nodes) == 1:
            tour.append(available_nodes[0])
            break

    return tour
