# Project 2 - Chatbot Project
# bot_model.py file: contains definition for neural network class used to train chatbot model using pytorch nn.Module
import torch
import torch.nn


class BotNN(torch.nn.Module):
    def __init__(self, input_layer_count, hidden_layer_count, class_count):
        super(BotNN, self).__init__()
        self.layer1 = torch.nn.Linear(input_layer_count, hidden_layer_count)
        self.layer2 = torch.nn.Linear(hidden_layer_count, hidden_layer_count)
        self.layer3 = torch.nn.Linear(hidden_layer_count, class_count)
        # Applying Rectified Linear (ReLU) activation function
        self.activation_function = torch.nn.ReLU()

    def forward(self, x):
        x = x.float()  # Convert  to float
        output = self.layer1(x)
        output = self.activation_function(output)
        output = self.layer2(output)
        output = self.activation_function(output)
        output = self.layer3(output)
        # cross entropy loss applies final activation function so not using it after layer 3
        return output
