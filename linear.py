import torch
import torch.nn.functional as F
import torch.nn as nn
import math

class Linear:
    def __init__(self,activation_type="Relu",intialization_type = "He"):
        # Initialize weights and biases
        self.w1 = torch.randn(784, 32, requires_grad=False)
        self.b1 = torch.zeros(32, requires_grad=False)
        self.w2 = torch.randn(32, 32, requires_grad=False)
        self.b2 = torch.zeros(32, requires_grad=False)
        self.w3 = torch.randn(32, 32, requires_grad=False)
        self.b3 = torch.zeros(32, requires_grad=False)
        self.w4 = torch.randn(32, 32, requires_grad=False)
        self.b4 = torch.zeros(32, requires_grad=False)
        self.w5 = torch.randn(32, 10, requires_grad=False)
        self.b5 = torch.zeros(10, requires_grad=False)

        if activation_type == "ReLU":
            self.activation_function = F.relu
            self.activation_function_derivative = self.relu_derivative
        if activation_type == "Sigmoid":
            self.activation_function = F.sigmoid
            self.activation_function_derivative = self.sigmoid_derivative
        if intialization_type == "He": self._he_initialize_weights()

    # He - initalization
    def _he_initialize_weights(self):
        for weight, bias in [(self.w1, self.b1), (self.w2, self.b2), (self.w3, self.b3), (self.w4, self.b4), (self.w5, self.b5)]:
            n = weight.size(1)
            bound = math.sqrt(2/n)
            nn.init.normal_(weight, mean=0, std=bound)
            nn.init.zeros_(bias)

    def softmax(self, x, dim=1):
        e_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)  # Subtracting max(x) for numerical stability
        return e_x / torch.sum(e_x, dim=dim, keepdim=True)

    def forward(self, x):
        # Forward pass for each layer
        self.a1 = x @ self.w1 + self.b1
        self.h1 = self.activation_function(self.a1)

        self.a2 = self.h1 @ self.w2 + self.b2
        self.h2 = self.activation_function(self.a2)

        self.a3 = self.h2 @ self.w3 + self.b3
        self.h3 = self.activation_function(self.a3)

        self.a4 = self.h3 @ self.w4 + self.b4
        self.h4 = self.activation_function(self.a4)

        self.a5 = self.h4 @ self.w5 + self.b5
        self.y_pred_prob = self.softmax(self.a5, dim=1)
        return self.y_pred_prob

    def compute_loss(self, y_pred, y_true):
        eps = 1e-6
        return -torch.sum(y_true * torch.log(y_pred + eps)) / y_true.shape[0]

    def backward(self, x, y_true, y_pred, learning_rate):
        # Output layer
        d_a5 = y_pred - y_true
        d_w5 = self.h4.t() @ d_a5
        d_b5 = d_a5.sum(0)

        # Layer 4
        d_h4 = d_a5 @ self.w5.t()
        d_h4 *= self.activation_function_derivative(self.a4)
        d_w4 = self.h3.t() @ d_h4
        d_b4 = d_h4.sum(0)

        # Layer 3
        d_h3 = d_h4 @ self.w4.t()
        d_h3 *= self.activation_function_derivative(self.a3)
        d_w3 = self.h2.t() @ d_h3
        d_b3 = d_h3.sum(0)

        # Layer 2
        d_h2 = d_h3 @ self.w3.t()
        d_h2 *= self.activation_function_derivative(self.a2)
        d_w2 = self.h1.t() @ d_h2
        d_b2 = d_h2.sum(0)

        # Layer 1
        d_h1 = d_h2 @ self.w2.t()
        d_h1 *= self.activation_function_derivative(self.a1)
        d_w1 = x.t() @ d_h1
        d_b1 = d_h1.sum(0)

        # Update weights and biases
        with torch.no_grad():
            self.w1 -= learning_rate * d_w1
            self.b1 -= learning_rate * d_b1
            self.w2 -= learning_rate * d_w2
            self.b2 -= learning_rate * d_b2
            self.w3 -= learning_rate * d_w3
            self.b3 -= learning_rate * d_b3
            self.w4 -= learning_rate * d_w4
            self.b4 -= learning_rate * d_b4
            self.w5 -= learning_rate * d_w5
            self.b5 -= learning_rate * d_b5

            # Zero gradients (manually, since we're not using autograd)
            self.zero_gradients()

    def sigmoid_derivative(self, x):
        return F.sigmoid(x) * (1 - F.sigmoid(x))

    def relu_derivative(self, x):
        return torch.where(x > 0, torch.tensor(1.0), torch.tensor(0.0))

    def zero_gradients(self):
        # Manually zero the gradients
        for param in [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w4, self.b4, self.w5, self.b5]:
            if param.grad is not None:
                param.grad.zero_()