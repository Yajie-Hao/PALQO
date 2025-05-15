import numpy as np
import quimb.tensor as qtn

# Quantum circuit construction functions
def apply_single_qubit_layer(circuit, gate_round=None):
    """Applies a single-qubit rotation layer."""
    for i in range(circuit.N):
        params = np.random.uniform(0, 2 * np.pi, 2)
        circuit.apply_gate("RY", params[0], i, gate_round=gate_round, parametrize=True)
        circuit.apply_gate("RZ", params[1], i, gate_round=gate_round, parametrize=True)

def apply_two_qubit_layer(circuit, gate="CZ", gate_round=None):
    """Applies a two-qubit entanglement layer."""
    for i in range(circuit.N - 1):
        circuit.apply_gate(gate, i, i + 1, gate_round=gate_round)

def create_ansatz_circuit(num_qubits, depth, entangling_gate="CZ", **kwargs):
    """Constructs a parameterized quantum circuit."""
    circuit = qtn.Circuit(num_qubits, **kwargs)
    for r in range(depth):
        apply_single_qubit_layer(circuit, gate_round=r)
        apply_two_qubit_layer(circuit, gate=entangling_gate, gate_round=r)
    return circuit

# Loss function
def calculate_energy(state, hamiltonian):
    """Calculates the energy expectation value."""
    bra, h, ket = qtn.tensor_network_align(state.H, hamiltonian, state)
    energy_tn = bra | h | ket
    return (energy_tn ^ ...).real

# PQC simulation and gradient calculation
def simulate_pqc_and_get_gradient(num_qubits, depth, ising_j, ising_bx):
    """Constructs PQC, defines loss function and optimizer, and returns gradient calculation function and number of parameters."""
    hamiltonian = qtn.MPO_ham_ising(num_qubits, ising_j, ising_bx)
    circuit = create_ansatz_circuit(num_qubits, depth, entangling_gate="CZ")

    tnopt = qtn.TNOptimizer(
        circuit.psi,  # the tensor network we want to optimize
        calculate_energy,  # the function we want to minimize
        loss_constants={"hamiltonian": hamiltonian}, 
        autodiff_backend="jax",
        optimizer="sgd",  # the optimization algorithm
        tags=["RZ", "RY"],
    )
    value_and_grad_fn = tnopt.vectorized_value_and_grad
    num_params = tnopt.vectorizer.d
    return value_and_grad_fn, num_params

# Objective function (energy and gradient)
def energy_and_gradient(params, pqc_fn, energy_history=None):
    """Calculates energy and gradient, and can record energy history."""
    energy, gradient = pqc_fn(params)
    if energy_history is not None:
        energy_history.append(energy)
        if len(energy_history) % 10 == 0:
            print(f"Step: {len(energy_history)},\tEnergy: {energy}")
    return energy, gradient

# Gradient descent optimizer
def gradient_descent_optimizer(initial_params, pqc_fn, learning_rate, max_iterations):
    """Performs gradient descent optimization."""
    params = initial_params.copy()
    param_history = []
    energy_history_temp = []
    for i in range(max_iterations):
        energy, gradient = energy_and_gradient(params, pqc_fn, energy_history_temp)
        params = params - learning_rate * gradient
        param_history.append(np.concatenate(([0.01 * (i + 1)], params.copy()), axis=0))
        # Convergence condition can be added, e.g., gradient norm less than tolerance
    param_history_array = np.array(param_history[:-1])
    energy_history_array = np.array([energy for _, energy in enumerate(energy_history_temp)])
    output_history = param_history_array.copy()
    output_history[:, 0] = energy_history_array[:-1]
    np.save("Ising_data.npy", param_history_array)
    np.save("Ising_data_y.npy", output_history)
    return params, energy_history_temp

# PQC gradient optimization
def optimize_pqc_gradient(initial_params, learning_rate, max_iterations, fun_pqc):
    """Optimizes PQC parameters using gradient descent."""
    optimal_params, energy_history = gradient_descent_optimizer(
        initial_params, fun_pqc, learning_rate, max_iterations
    )
    return optimal_params, energy_history

# Actual experiment (calls JAX to calculate energy and gradient)
def perform_real_experiment(params, fun_pqc):
    """Performs the actual quantum experiment (simulator), calculating energy and gradient."""
    energy, gradient = fun_pqc(params)
    return energy, gradient

# Get training data
def get_training_data(num_data=None):
    """Loads training data."""
    data = np.load("Ising_data.npy")[:num_data]
    data_y = np.load("Ising_data_y.npy")[:num_data]
    return data, data_y