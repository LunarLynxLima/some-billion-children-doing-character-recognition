import math
import operations as ops

import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear:
    def __init__(self, inputs = 784, hidden = [32, 32], outputs = 10, activation_type = "ReLU", initialization_type = "He", custom_array = True):
        self.layers = []
        requires_grad = False
        self.custom_array = custom_array
        
        if len(hidden) == 0: hidden = [outputs]

        w1 = torch.randn(inputs, hidden[0], requires_grad=requires_grad)
        b1 = torch.zeros(hidden[0], requires_grad=requires_grad)
        self.layers.append([w1, b1])
        for i in range(1, len(hidden)):
            w = torch.randn(hidden[i-1], hidden[i], requires_grad=requires_grad)
            b = torch.zeros(hidden[i], requires_grad=requires_grad)
            self.layers.append([w, b])
        
        w_last = torch.randn(hidden[-1], outputs, requires_grad=requires_grad)
        b_last = torch.zeros(outputs, requires_grad=requires_grad)
        self.layers.append([w_last, b_last])

        self.num_layers = len(self.layers)

        if initialization_type == "He":  self._he_initialize_weights()
        elif initialization_type == "Xavier": self._xavier_initialize_weights()
        else: raise ValueError("Invalid initialization type")

        if activation_type == "ReLU": 
            self.activation_function = self.relu
            self.activation_function_derivative = self.relu_derivative
        elif activation_type == "Sigmoid": 
            self.activation_function = self.sigmoid
            self.activation_function_derivative = self.sigmoid_derivative
        else: raise ValueError("Invalid activation function")

        if self.custom_array:
            for i in range(len(self.layers)):
                self.layers[i][0] = ops.CustomArray(self.layers[i][0])
                self.layers[i][1] = ops.CustomArray(self.layers[i][1])
        
    def forward(self, x):
        self.activations, self.z_values = [], []

        a = x
        for w, b in self.layers[:-1]:
            if self.custom_array: 
                a = ops.CustomArray(a)
                z = ops.CustomArray(a @ w + b)
                a = self.activation_function(z) 
                self.activations.append(a)
                self.z_values.append(z)
            else:
                z = a @ w + b
                a = self.activation_function(z)
                self.z_values.append(z)
                self.activations.append(a)
        
        w_last, b_last = self.layers[-1]
        z_last = a @ w_last + b_last

        if self.custom_array:  z_last = ops.CustomArray(z_last)

        self.z_values.append(z_last)
        self.y_pred_prob = self.softmax(z_last, dim=1)

        # print(self.y_pred_prob)
        return self.y_pred_prob

    def backward(self, x, y_true, y_pred, learning_rate):
        if self.custom_array: y_true = ops.CustomArray._convert_to_custom_numbers_array(y_true)  

        gradients = {}

        # Output layer gradients   
        d_a = y_pred - y_true # custom array

        if self.custom_array:
            if self.activations:
                torch_acti_transpose = ops.CustomArray._transpose(self.activations[-1])
                print(type(torch_acti_transpose))
                d_w = torch_acti_transpose @ d_a
            else:
                torch_x_transpose = ops.CustomArray._transpose(x)
                d_w = torch_x_transpose @ d_a
            d_a = ops.CustomArray._convert_to_torch_tensor(d_a)  # update this to have custom array sum directly
            d_b = d_a.sum(0)

            d_w, d_b = ops.CustomArray(d_w), ops.CustomArray(d_b)
            gradients[f"d_w{self.num_layers}"], gradients[f"d_b{self.num_layers}"] = d_w, d_b

        else:
            d_w = self.activations[-1].t() @ d_a if self.activations else x.t() @ d_a
            d_b = d_a.sum(0)
            gradients[f"d_w{self.num_layers}"], gradients[f"d_b{self.num_layers}"] = d_w, d_b
            # if self.activations:
            #     d_w = self.activations[-1].t() @ d_a
            # else:
            #     d_w = x.t() @ d_a
            # d_b = d_a.sum(0)

        # Backpropagate through hidden layers
        for i in range(self.num_layers - 2, -1, -1):
            if self.custom_array:
                print(f"Custom Array {i}")
                z_transpose = ops.CustomArray._transpose(self.layers[i+1][0])
                print("passed")
                d_a = d_a @ z_transpose * self.activation_function_derivative(self.z_values[i])
                print("passed")
                if i > 0:
                    acti_transpose = ops.CustomArray._transpose(self.activations[i-1])
                    d_w = acti_transpose @ d_a
                else:
                    x_transpose = ops.CustomArray._transpose(x)
                    d_w = x_transpose @ d_a
                d_a = ops.CustomArray._convert_to_torch_tensor(d_a)
                d_b = d_a.sum(0)

                d_w, d_b = ops.CustomArray(d_w), ops.CustomArray(d_b)
                gradients[f"d_w{i+1}"], gradients[f"d_b{i+1}"] = d_w, d_b


            else: 
                d_a = d_a @ self.layers[i+1][0].t() * self.activation_function_derivative(self.z_values[i])
                # d_w = self.activations[i-1].t() @ d_a if i > 0 else x.t() @ d_a
                if i > 0:
                    d_w = self.activations[i-1].t() @ d_a
                else:
                    d_w = x.t() @ d_a
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
    def _xavier_initialize_weights(self):
        for weight, bias in self.layers:
            n = weight.size(1)
            bound = math.sqrt(6 / n)
            nn.init.uniform_(weight, -bound, bound)
            nn.init.zeros_(bias)


    def softmax(self, x, dim=1):
        if self.custom_array :
            x = ops.CustomArray._convert_to_torch_tensor(x)
            e_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
            return ops.CustomArray(e_x / torch.sum(e_x, dim=dim, keepdim=True))
        
        e_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
        return (e_x / torch.sum(e_x, dim=dim, keepdim=True))
    

    def relu(self, x):
        if self.custom_array: 
            x = ops.CustomArray._convert_to_torch_tensor(x)
            return ops.CustomArray(F.relu(x))
        return F.relu(x)
    def relu_derivative(self, x): 
        if self.custom_array: 
            x = ops.CustomArray._convert_to_torch_tensor(x)
            return ops.CustomArray(torch.where(x > 0, torch.tensor(1.0), torch.tensor(0.0)))
        return torch.where(x > 0, torch.tensor(1.0), torch.tensor(0.0))
    
    def sigmoid(self, x):
        if self.custom_array: 
            x = ops.CustomArray._convert_to_torch_tensor(x)
            return ops.CustomArray(torch.sigmoid(x))
        return torch.sigmoid(x)
    def sigmoid_derivative(self, x):
        if self.custom_array: 
            x = ops.CustomArray._convert_to_torch_tensor(x)
            sig = torch.sigmoid(x)
            return ops.CustomArray(sig * (1 - sig))
        sig = torch.sigmoid(x)
        return sig * (1 - sig)
    
    def compute_loss(self, y_pred, y_true, eps=1e-6): 
        if self.custom_array: 
            y_pred = ops.CustomArray._convert_to_torch_tensor(y_pred)
            return ops.CustomArray((-torch.sum(y_true * torch.log(y_pred + eps)) / y_true.shape[0]))
        return (-torch.sum(y_true * torch.log(y_pred + eps)) / y_true.shape[0])
    
    def save_backward(self, x, y_true, y_pred, learning_rate):
        if self.custom_array: 
            y_true = ops.CustomArray._convert_to_custom_numbers_array(y_true)  

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