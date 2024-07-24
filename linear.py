import math
import torch
import torch.nn.functional as F
import torch.nn as nn

class Linear:
    def __init__(self, inputs=784, hidden=[], outputs=10, activation_type="ReLU", initialization_type="He"):
        self.layers = []
        
        if len(hidden) == 0:
            w1 = torch.randn(inputs, outputs, requires_grad=True)
            b1 = torch.zeros(outputs, requires_grad=True)
            self.layers.append([w1, b1])
        else:
            w1 = torch.randn(inputs, hidden[0], requires_grad=True)
            b1 = torch.zeros(hidden[0], requires_grad=True)
            self.layers.append([w1, b1])
            for i in range(1, len(hidden)):
                w = torch.randn(hidden[i-1], hidden[i], requires_grad=True)
                b = torch.zeros(hidden[i], requires_grad=True)
                self.layers.append([w, b])
            
            w_last = torch.randn(hidden[-1], outputs, requires_grad=True)
            b_last = torch.zeros(outputs, requires_grad=True)
            self.layers.append([w_last, b_last])

        self.num_layers = len(self.layers)
        if initialization_type == "He":  self._he_initialize_weights()
        else: raise ValueError("Invalid initialization type")
        if activation_type == "ReLU": 
            self.activation_function = F.relu
            self.activation_function_derivative = self.relu_derivative
        elif activation_type == "Sigmoid": 
            self.activation_function = torch.sigmoid
            self.activation_function_derivative = self.sigmoid_derivative
        else:
            raise ValueError("Invalid activation function")
        
    def forward(self, x):
        self.activations, self.z_values = [], []

        a = x
        for w, b in self.layers[:-1]:
            z = a @ w + b
            a = self.activation_function(z)
            self.z_values.append(z)
            self.activations.append(a)
        
        w_last, b_last = self.layers[-1]
        z_last = a @ w_last + b_last
        self.z_values.append(z_last)
        self.y_pred_prob = self.softmax(z_last, dim=1)
        
        return self.y_pred_prob

    def backward(self, x, y_true, y_pred, learning_rate):
        gradients = {}
        
        # Output layer gradients
        d_a = y_pred - y_true
        d_w = self.activations[-1].t() @ d_a if self.activations else x.t() @ d_a
        d_b = d_a.sum(0)

        gradients[f"d_w{self.num_layers}"], gradients[f"d_b{self.num_layers}"] = d_w, d_b

        # Backpropagate through hidden layers
        for i in range(self.num_layers - 2, -1, -1):
            d_a = d_a @ self.layers[i+1][0].t() * self.activation_function_derivative(self.z_values[i])
            d_w = self.activations[i-1].t() @ d_a if i > 0 else x.t() @ d_a
            d_b = d_a.sum(0)

            gradients[f"d_w{i+1}"], gradients[f"d_b{i+1}"] = d_w, d_b
        
        # Update weights and biases
        with torch.no_grad():
            for i in range(1, self.num_layers + 1):
                self.layers[i-1][0] -= learning_rate * gradients[f"d_w{i}"]
                self.layers[i-1][1] -= learning_rate * gradients[f"d_b{i}"]

    def zero_gradients(self):
        for params in self.layers:
            if params[0].grad is not None: params[0].grad.zero_()
            if params[1].grad is not None: params[1].grad.zero_()

    def _he_initialize_weights(self):
        for weight, bias in self.layers:
            n = weight.size(1)
            bound = math.sqrt(2 / n)
            nn.init.normal_(weight, mean=0, std=bound)
            nn.init.zeros_(bias)

    def softmax(self, x, dim=1):
        e_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
        return e_x / torch.sum(e_x, dim=dim, keepdim=True)

    def sigmoid_derivative(self, x):
        sig = torch.sigmoid(x)
        return sig * (1 - sig)
    def relu_derivative(self, x): return torch.where(x > 0, torch.tensor(1.0), torch.tensor(0.0))
    def compute_loss(self, y_pred, y_true, eps=1e-6): return -torch.sum(y_true * torch.log(y_pred + eps)) / y_true.shape[0]