import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset
import quimb.tensor as qtn
from torch.autograd import Variable, grad
import time

start_time = time.time()
# Set visible CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

# Define device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# PQC parameter list
qubit_epoch_list = (20, 3, -1, 2)
(
    num_qubits,
    pqc_depth,
    ising_j_coupling,
    ising_b_field_coeff,
) = qubit_epoch_list

# Ising Hamiltonian parameters
ISING_J = ising_j_coupling * 4
ISING_BX = -1 * ising_j_coupling / ising_b_field_coeff * 2


# VQE optimizer parameters
VQE_LEARNING_RATE = 0.01
VQE_MAX_ITER = 20
VQE_INIT_SEED = 2025

# Net (neural network) training parameters
NET_SEED = 2025
NET_HIDDEN_DIM = 50 * 2 * num_qubits * pqc_depth
NET_LEARNING_RATE = 1e-4
NET_WEIGHT_DECAY = 0.00001
NET_LOSS_THRESHOLD = 5e-6 
NET_NUM_EPOCH = 3000

# PALQO iteration parameters
PALQO_SAMPLE = 2
PALQO_MAX_ITER = VQE_MAX_ITER - PALQO_SAMPLE
PALQO_PRE_NUM = 2000

# Quantum circuit construction function
def apply_single_qubit_layer(circuit, gate_round=None):
    """Applies a single-qubit rotation layer."""
    for i in range(circuit.N):
        # Use numpy.random.uniform for conciseness
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
def simulate_pqc_and_get_gradient(num_qubits, depth):
    """Constructs PQC, defines loss function and optimizer, and returns gradient calculation function and number of parameters."""
    hamiltonian = qtn.MPO_ham_ising(num_qubits, ISING_J, ISING_BX)
    circuit = create_ansatz_circuit(num_qubits, depth, entangling_gate="CZ")

    tnopt = qtn.TNOptimizer(
        circuit.psi,  # the tensor network we want to optimize
        calculate_energy,  # the function we want to minimize
        loss_constants={"hamiltonian": hamiltonian},  # supply U to the loss function as a constant TN
        autodiff_backend="jax",
        optimizer="sgd",  # the optimization algorithm
        tags=["RZ", "RY"],
    )
    value_and_grad_fn = tnopt.vectorized_value_and_grad
    num_params = tnopt.vectorizer.d
    return value_and_grad_fn, num_params



# Initialize PQC model
fun_pqc, num_pqc_params = simulate_pqc_and_get_gradient(num_qubits, pqc_depth)
print("JAX compiling...")
fun_pqc(VQE_INIT_SEED)  # First call triggers JAX compilation
print("JAX compiled.")



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
    # param_history = [np.concatenate(([0.01 * (i + 1)], params.copy()), axis=0)]
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
    # param_history_array = np.array(param_history[:-1])
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



# VQE baseline experiment
print("-" * 60 + "VQE-baseline" + "-" * 60)
np.random.seed(VQE_INIT_SEED)
initial_pqc_params = np.random.uniform(0, 1, num_pqc_params)
optimal_pqc_params_vqe, vqe_energy_history = optimize_pqc_gradient(
    initial_pqc_params, VQE_LEARNING_RATE, VQE_MAX_ITER, fun_pqc
)
print("The VQE energy list is", vqe_energy_history)
print("-" * 60 + "VQE-baseline Finished" + "-" * 60)

# PALQO iteration start
print("-" * 60 + "PALQO Iteration Start" + "-" * 60)



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



# Get prediction from Net and evaluate
def get_predicted_and_evaluate(index, training_data, neural_net):
    """Uses the trained neural network to predict subsequent parameters and evaluates the energy."""
    time = (index + 1) * 0.01
    current_params_with_time = training_data[index]
    predicted_params_with_time = current_params_with_time.copy()
    predicted_thetas = dict()
    speed_factor = 1
    num_predictions = PALQO_PRE_NUM

    for i in range(num_predictions):
        input_params = predicted_params_with_time[1:]  # Exclude the time step
        predicted_thetas[i] = input_params
        input_tensor = torch.Tensor(input_params).unsqueeze(0).to(device)
        time_tensor = torch.Tensor([[time]]).to(device)
        input_tensor.requires_grad = True
        time_tensor.requires_grad = True
        output = neural_net(time_tensor, input_tensor)
        predicted_params_with_time = output.clone().cpu().detach().numpy()[0]
        time = (index + i * speed_factor + 2) * 0.01

    best_energy, _ = perform_real_experiment(predicted_thetas[num_predictions-1], fun_pqc)
    best_theta = predicted_thetas[num_predictions-1]
    return best_theta, best_energy



# Set random seed
def set_seed(seed):
    """Sets all relevant random seeds to ensure experiment reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



# Define neural network model
class ParameterPredictorNet(nn.Module):
    """Neural network to predict PQC parameters."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ParameterPredictorNet, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, time, params):
        inputs = torch.cat([time, params], dim=1)
        out = torch.tanh(self.input_layer(inputs))
        out = torch.tanh(self.hidden_layer1(out))
        out = torch.tanh(self.hidden_layer2(out))
        return self.output_layer(out)



# Initialize neural network
set_seed(NET_SEED)
net = ParameterPredictorNet(num_pqc_params + 1, NET_HIDDEN_DIM, num_pqc_params + 1).to(device)
optimizer_net = torch.optim.Adam(net.parameters(), lr=NET_LEARNING_RATE, weight_decay=NET_WEIGHT_DECAY)



# Define PINN loss functions
def pde_theta_loss(predicted_output, true_params_with_time, learning_rate_vqe):
    """Loss for parameter theta in PINN."""
    epsilon, theta_next = predicted_output[:, 0:1], predicted_output[:, 1:]
    epsilon_theta = torch.autograd.grad(epsilon.sum(), true_params_with_time, create_graph=False, retain_graph=True)[0]
    pde_residual = theta_next - true_params_with_time + learning_rate_vqe * epsilon_theta
    return torch.mean(pde_residual ** 2)



def pde_epsilon_loss(time_tensor, true_params_with_time, predicted_output, learning_rate_vqe):
    """Loss for epsilon in PINN."""
    epsilon = predicted_output[:, 0:1]
    epsilon_t = torch.autograd.grad(epsilon.sum(), time_tensor, create_graph=False, retain_graph=True)[0]
    epsilon_theta = torch.autograd.grad(epsilon.sum(), true_params_with_time, create_graph=False, retain_graph=True)[0]
    pde_term = epsilon_t + learning_rate_vqe * (epsilon_theta ** 2).sum(dim=1, keepdim=True)
    return torch.mean(pde_term ** 2)



def net_output_loss(predicted_output, target_output):
    """Loss between neural network output and target output."""
    output_loss = predicted_output - target_output
    return torch.mean(output_loss ** 2)



# Train neural network
def train_neural_network(num_train_data, current_iteration):
    """Trains the neural network to predict PQC parameters."""
    set_seed(NET_SEED)
    for layer in net.modules():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()

    if current_iteration == 0:
        training_data, target_data = get_training_data(num_train_data)
    else:
        training_data, target_data = get_training_data(num_train_data)
        training_data = training_data[int(-1 * PALQO_SAMPLE) :]
        target_data = target_data[int(-1 * PALQO_SAMPLE) :]

    time_tensor = torch.from_numpy(training_data[:, :1]).float().to(device)
    params_tensor = torch.from_numpy(training_data[:, 1:]).float().to(device)
    target_tensor = torch.from_numpy(target_data).float().to(device)

    dataset = TensorDataset(time_tensor, params_tensor, target_tensor)
    dataloader = DataLoader(dataset, batch_size=PALQO_SAMPLE, shuffle=False)

    def lr_schedule(step):
        total_steps = NET_NUM_EPOCH
        return (total_steps - step) / total_steps

    scheduler = LambdaLR(optimizer_net, lr_lambda=lr_schedule)
    losses = []

    for epoch in range(NET_NUM_EPOCH):
        for time_batch, params_batch, target_batch in dataloader:
            optimizer_net.zero_grad()
            time_batch = Variable(time_batch.float(), requires_grad=True)
            params_batch = Variable(params_batch.float(), requires_grad=True)
            target_batch = Variable(target_batch.float(), requires_grad=True)

            predicted_output = net(time_batch, params_batch)

            theta_loss = pde_theta_loss(predicted_output, params_batch, learning_rate_vqe=VQE_LEARNING_RATE)
            epsilon_loss = pde_epsilon_loss(
                time_batch, params_batch, predicted_output, learning_rate_vqe=VQE_LEARNING_RATE
            )
            output_loss = 0.0001 * net_output_loss(predicted_output, target_batch)

            loss = theta_loss + epsilon_loss + output_loss
            loss.backward()
            optimizer_net.step()
            scheduler.step()
            losses.append(loss.item())

        if (epoch + 1) % 1000 == 0:
            print(
                f"Epoch {epoch + 1}, Training Loss: {loss.item():.6f}, LR: {optimizer_net.param_groups[0]['lr']:.6e}"
            )
            print(
                f"Theta Loss: {theta_loss.item():.6f}, Epsilon Loss: {epsilon_loss.item():.6f}, Output Loss: {output_loss.item():.6f}"
            )
            end_time = time.time()

            run_time = end_time - start_time
            print(f"Runtime: {run_time:.4f} s")
        if loss < NET_LOSS_THRESHOLD:
            print("Training stopped: learning rate too low and loss sufficiently small.")
            test_inputs, _ = get_training_data()  # Changed get_test_data() to get_training_data()

            index = 0
            best_theta_list_op, best_min_value = get_predicted_and_evaluate(index, test_inputs, net)

            return best_theta_list_op, best_min_value

        if (epoch + 1) % NET_NUM_EPOCH == 0:
            test_inputs, _ = get_training_data() # Changed get_test_data() to get_training_data()
            index = 0
            best_theta_list_op, best_min_value = get_predicted_and_evaluate(index, test_inputs, net)
            return best_theta_list_op, best_min_value
    return best_theta_list_op, best_min_value



theta_list_op_temp = initial_pqc_params.tolist().copy()
iter_pinn = 0
PALQO_energy_list = []
PALQO_energy_list.extend(vqe_energy_history[:PALQO_SAMPLE]) # Changed f_list to vqe_energy_history
num_train_data = PALQO_SAMPLE
repeat = False
f_list_temp = vqe_energy_history[:PALQO_SAMPLE].copy() # Changed f_list to vqe_energy_history

for iter in range(PALQO_MAX_ITER):
    print("-" * 60 + f"PALQO Iteration -{iter}" + "-" * 60)
    theta_list_op, temp_min_value = train_neural_network(num_train_data, iter_pinn)
    print("num_train_data", num_train_data)
    if temp_min_value < f_list_temp[-1]:
        initial_pqc_params = theta_list_op
        iter_pinn += 1
        if iter >= 1:
            PALQO_energy_list.extend(f_list_temp[:num_train_data])
        num_train_data = PALQO_SAMPLE
        theta_list_op_temp = theta_list_op.copy()
        repeat = False
    else:
        initial_pqc_params = theta_list_op_temp
        num_train_data = int(num_train_data + 1)
        repeat = True
    if repeat == False and (iter_pinn + 1) == 2:
        num_train_data = PALQO_SAMPLE
    max_iter = num_train_data + 1
    print("The PALQO_energy_list is", PALQO_energy_list)

    x_opt_temp, f_list_temp = optimize_pqc_gradient(initial_pqc_params, VQE_LEARNING_RATE, max_iter, fun_pqc)

if repeat == True:
    PALQO_energy_list.extend(f_list_temp[:num_train_data])

print("The PALQO_energy_list is", PALQO_energy_list)
