import math
import torch
import torch.nn.functional as F
import torch.nn as nn

class Linear:
    def __init__(self,inputs = 784, hidden = [32,], outputs = 10, activation_type="Relu",intialization_type = "He"):
        # Initialize weights and biases
        self.layers = []
        if len(hidden) == 0:
            self.w1 = torch.randn(inputs, outputs, requires_grad=False)
            self.b1 = torch.zeros(outputs, requires_grad=False)
            self.layers.append([self.w1, self.b1])
        else:
            self.w1 = torch.randn(inputs, hidden[0], requires_grad=False)
            self.b1 = torch.zeros(hidden[0], requires_grad=False)
            self.layers.append([self.w1, self.b1])
            for i in range(len(hidden)):
                setattr(self, f"w{i+2}", torch.randn(hidden[i-1], hidden[i], requires_grad=False))
                setattr(self, f"b{i+2}", torch.zeros(hidden[i], requires_grad=False))
                self.layers.append([getattr(self, f"w{i+2}"), getattr(self, f"b{i+2}")])
            
            setattr(self, f"w{len(hidden)+2}", torch.randn(hidden[-1], outputs, requires_grad=False))
            setattr(self, f"b{len(hidden)+2}", torch.zeros(outputs, requires_grad=False))
            self.layers.append([getattr(self, f"w{len(hidden)+2}"), getattr(self, f"b{len(hidden)+2}")])

        self.num_layers = (len(self.layers)) # len(hidden) + 1 

        if activation_type == "ReLU": 
            self.activation_function = F.relu
            self.activation_function_derivative = self.relu_derivative
        elif activation_type == "Sigmoid": 
            self.activation_function = F.sigmoid
            self.activation_function_derivative = self.sigmoid_derivative
        else:
            raise ValueError("Invalid activation function")
        if intialization_type == "He": self._he_initialize_weights()


    def forward(self, x):
        # Forward pass for each layer
        self.a1 = x @ self.w1 + self.b1
        self.h1 = self.activation_function(self.a1)

        for i in range(len(self.layers[1:self.num_layers-1])):
            setattr(self, f"a{i+2}", getattr(self, f"h{i+1}") @ self.layers[i+1][0] + self.layers[i+1][1])
            setattr(self, f"h{i+2}", self.activation_function(getattr(self, f"a{i+2}")))

        if len(self.layers) == 1:
            self.a2 = getattr(self, f"h{len(self.layers)}") @ self.layers[-1][0] + self.layers[-1][1]
            self.y_pred_prob = self.softmax(self.a2, dim=1)
        else:
            setattr(self, f"a{len(self.layers)}", getattr(self, f"h{len(self.layers) - 1}") @ self.layers[-1][0] + self.layers[-1][1])
            self.y_pred_prob = self.softmax(getattr(self, f"a{len(self.layers)}"), dim=1)

        return self.y_pred_prob

    #! TODO : Implement backward for variable layers
    def backward(self, x, y_true, y_pred, learning_rate):
        gradients = {}

        # Output layer
        gradients[f"d_a{self.num_layers}"] = y_pred - y_true
        gradients[f"d_w{self.num_layers}"] = getattr(self, f"h{self.num_layers-1}").t() @ gradients[f"d_a{self.num_layers}"]
        gradients[f"d_b{self.num_layers}"] = gradients[f"d_a{self.num_layers}"].sum(0)

        # Hidden layers
        for i in range(self.num_layers-1, 0, -1):
            if i == self.num_layers-1:
                gradients[f"d_h{i}"] = gradients[f"d_a{i+1}"] @ getattr(self, f"w{i+1}").t()
                gradients[f"d_h{i}"] *= self.activation_function_derivative(getattr(self, f"a{i}"))
                gradients[f"d_w{i}"] = getattr(self, f"h{i-1}").t() @ gradients[f"d_h{i}"]
                gradients[f"d_b{i}"] = gradients[f"d_h{i}"].sum(0)
            elif i != 1 and i != self.num_layers-1:
                gradients[f"d_h{i}"] = gradients[f"d_h{i+1}"] @ getattr(self, f"w{i+1}").t()
                gradients[f"d_h{i}"] *= self.activation_function_derivative(getattr(self, f"a{i}"))
                gradients[f"d_w{i}"] = getattr(self, f"h{i-1}").t() @ gradients[f"d_h{i}"]
                gradients[f"d_b{i}"] = gradients[f"d_h{i}"].sum(0)
            elif i == 1:
                gradients[f"d_h{i}"] = gradients[f"d_h{i+1}"] @ getattr(self, f"w{i+1}").t()
                gradients[f"d_h{i}"] *= self.activation_function_derivative(getattr(self, f"a{i}"))
                gradients[f"d_w{i}"] = x.t() @ gradients[f"d_h{i}"]
                gradients[f"d_b{i}"] = gradients[f"d_h{i}"].sum(0)
            else: 
                raise ValueError("It went tits up, Invalid layer number")
        
        # Update the weights and biases
        with torch.no_grad():
            for i in range(1, self.num_layers + 1):
                setattr(self, f"w{i}", getattr(self, f"w{i}") - (learning_rate * gradients[f"d_w{i}"]))
                setattr(self, f"b{i}", getattr(self, f"b{i}") - (learning_rate * gradients[f"d_b{i}"]))
        
            # Zero the gradients since we're not using autograd
            self.zero_gradients()

    def zero_gradients(self):
        # Manually zero the gradients
        for params in self.layers:
            if params[0].grad is not None: params[0].grad.zero_()
            if params[1].grad is not None: params[1].grad.zero_()

        # He - initalization
    def _he_initialize_weights(self):
        for weight, bias in self.layers:
            n = weight.size(1)
            bound = math.sqrt(2/n)
            nn.init.normal_(weight, mean=0, std=bound)
            nn.init.zeros_(bias)
        # for weight, bias in [(self.w1, self.b1), (self.w2, self.b2), (self.w3, self.b3), (self.w4, self.b4), (self.w5, self.b5)]:
        #     n = weight.size(1)
        #     bound = math.sqrt(2/n)
        #     nn.init.normal_(weight, mean=0, std=bound)
        #     nn.init.zeros_(bias)

    def softmax(self, x, dim=1):
        e_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)  # Subtracting max(x) for numerical stability
        return e_x / torch.sum(e_x, dim=dim, keepdim=True)
    def sigmoid_derivative(self, x): return F.sigmoid(x) * (1 - F.sigmoid(x))
    def relu_derivative(self, x): return torch.where(x > 0, torch.tensor(1.0), torch.tensor(0.0))
    def compute_loss(self, y_pred, y_true, eps = 1e-6): return -torch.sum(y_true * torch.log(y_pred + eps)) / y_true.shape[0]