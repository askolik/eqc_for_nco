import copy
import cirq


def graph_encoding_circuit(edges, qubits, n_layers, data_params):
    circuit = cirq.Circuit()
    circuit += cirq.H.on_each(qubits)

    for layer in range(n_layers):
        edge_weights = data_params[layer][-1]
        for edge_ix, edge in enumerate(edges):
            circuit.append(
                cirq.CNOT(qubits[edge[0]],
                          qubits[edge[1]]))

            circuit.append(cirq.rz(edge_weights[edge_ix])(qubits[edge[1]]))

            circuit.append(
                cirq.CNOT(qubits[edge[0]],
                          qubits[edge[1]]))

        for qubit_ix, qubit in enumerate(qubits):
            circuit += cirq.rx(data_params[layer][qubit_ix])(qubit)

    # print(circuit)
    # exit()

    return circuit


def hardware_efficient_circuit(
        edges, qubits, n_layers, params, data_params, use_reuploading=False):

    n_qubits = len(qubits)
    circuit = cirq.Circuit()
    params = copy.deepcopy(params)

    for layer in range(n_layers):
        if layer == 0 or use_reuploading:
            for qubit_ix, qubit in enumerate(qubits):
                circuit += cirq.rx(data_params[layer][qubit_ix])(qubit)

            edge_weights = data_params[layer][-1]
            for edge_ix, edge in enumerate(edges):
                circuit.append(
                    cirq.CNOT(qubits[edge[0]],
                              qubits[edge[1]]))

                circuit.append(cirq.rz(edge_weights[edge_ix])(qubits[edge[1]]))

                circuit.append(
                    cirq.CNOT(qubits[edge[0]],
                              qubits[edge[1]]))

        for qubit_ix, qubit in enumerate(qubits):
            circuit += cirq.ry(params[layer].pop())(qubit)

        for i in range(0, n_qubits):
            circuit += cirq.CZ(qubits[i], qubits[(i + 1) % n_qubits])

    # print(circuit)
    # exit()

    return circuit
