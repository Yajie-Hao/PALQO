import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable, grad
import time
from vqa import get_training_data, perform_real_experiment

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

# Get prediction from Net and evaluate
def get_predicted_and_evaluate(index, training_data, neural_net, fun_pqc, palqo_pre_num):
    """Uses the trained neural network to predict subsequent parameters and evaluates the energy."""
    time = (index + 1) * 0.01
    current_params_with_time = training_data[index]
    predicted_params_with_time = current_params_with_time.copy()
    predicted_thetas = dict()
    speed_factor = 1
    num_predictions = palqo_pre_num

    for i in range(num_predictions):
        input_params = predicted_params_with_time[1:]  # Exclude the time step
        predicted_thetas[i] = input_params
        input_tensor = torch.Tensor(input_params).unsqueeze(0).to(neural_net.input_layer.weight.device)
        time_tensor = torch.Tensor([[time]]).to(neural_net.input_layer.weight.device)
        input_tensor.requires_grad = True
        time_tensor.requires_grad = True
        output = neural_net(time_tensor, input_tensor)
        predicted_params_with_time = output.clone().cpu().detach().numpy()[0]
        time = (index + i * speed_factor + 2) * 0.01

    best_energy, _ = perform_real_experiment(predicted_thetas[num_predictions-1], fun_pqc)
    best_theta = predicted_thetas[num_predictions-1]
    return best_theta, best_energy

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
def train_neural_network(net, optimizer_net, num_train_data, current_iteration, vqe_learning_rate, net_num_epoch, net_loss_threshold, palqo_sample, fun_pqc, palqo_pre_num):
    """Trains the neural network to predict PQC parameters."""
    set_seed(2025) # Use NET_SEED if you pass it
    for layer in net.modules():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()

    if current_iteration == 0:
        training_data, target_data = get_training_data(num_train_data)
    else:
        training_data, target_data = get_training_data(num_train_data)
        training_data = training_data[int(-1 * palqo_sample) :]
        target_data = target_data[int(-1 * palqo_sample) :]

    time_tensor = torch.from_numpy(training_data[:, :1]).float().to(net.input_layer.weight.device)
    params_tensor = torch.from_numpy(training_data[:, 1:]).float().to(net.input_layer.weight.device)
    target_tensor = torch.from_numpy(target_data).float().to(net.input_layer.weight.device)

    dataset = TensorDataset(time_tensor, params_tensor, target_tensor)
    dataloader = DataLoader(dataset, batch_size=palqo_sample, shuffle=False)

    def lr_schedule(step):
        total_steps = net_num_epoch
        return (total_steps - step) / total_steps

    scheduler = LambdaLR(optimizer_net, lr_lambda=lr_schedule)
    losses = []

    for epoch in range(net_num_epoch):
        for time_batch, params_batch, target_batch in dataloader:
            optimizer_net.zero_grad()
            time_batch = Variable(time_batch.float(), requires_grad=True)
            params_batch = Variable(params_batch.float(), requires_grad=True)
            target_batch = Variable(target_batch.float(), requires_grad=True)

            predicted_output = net(time_batch, params_batch)

            theta_loss = pde_theta_loss(predicted_output, params_batch, learning_rate_vqe=vqe_learning_rate)
            epsilon_loss = pde_epsilon_loss(
                time_batch, params_batch, predicted_output, learning_rate_vqe=vqe_learning_rate
            )
            output_loss = 0.0001 * net_output_loss(predicted_output, target_batch)

            loss = theta_loss + epsilon_loss + output_loss
            loss.backward()
            optimizer_net.step()
            scheduler.step()
            losses.append(loss.item())

        if (epoch + 1) % 3000 == 0:
            print(
                f"Epoch {epoch + 1}, Training Loss: {loss.item():.6f}, LR: {optimizer_net.param_groups[0]['lr']:.6e}"
            )
            print(
                f"Theta Loss: {theta_loss.item():.6f}, Epsilon Loss: {epsilon_loss.item():.6f}, Output Loss: {output_loss.item():.6f}"
            )

        if loss < net_loss_threshold:
            print("Training finished")
            test_inputs, _ = get_training_data()
            index = 0
            best_theta_list_op, best_min_value = get_predicted_and_evaluate(index, test_inputs, net, fun_pqc, palqo_pre_num)
            return best_theta_list_op, best_min_value

        if (epoch + 1) % net_num_epoch == 0:
            print("Training finished")
            test_inputs, _ = get_training_data()
            index = 0
            best_theta_list_op, best_min_value = get_predicted_and_evaluate(index, test_inputs, net, fun_pqc, palqo_pre_num)
            return best_theta_list_op, best_min_value
    return best_theta_list_op, best_min_value
