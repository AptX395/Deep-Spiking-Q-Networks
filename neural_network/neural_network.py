# coding = utf-8

import torch
from torch import nn, optim
from spikingjelly.clock_driven import neuron, surrogate, layer, functional


class Dqn(nn.Module):
    def __init__(self, history_len, action_num, learning_rate, device):
        super(Dqn, self).__init__()

        # Define the neural network layers.
        self._conv1 = nn.Sequential(nn.Conv2d(history_len, out_channels=32, kernel_size=8, stride=4),
                                    nn.ReLU())
        self._conv2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
                                    nn.ReLU())
        self._conv3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
                                    nn.ReLU())
        self._fc1 = nn.Sequential(nn.Flatten(),
                                  nn.Linear(in_features=64 * 7 * 7, out_features=512),
                                  nn.ReLU())
        self._fc2 = nn.Linear(in_features=512, out_features=action_num)

        self.device = device
        self.optimizer = optim.Adam(self.parameters(), learning_rate)
        self.loss_func = nn.SmoothL1Loss()
        self.to(self.device)

    def forward(self, x):
        x = self._conv1(x)
        x = self._conv2(x)
        x = self._conv3(x)
        x = self._fc1(x)
        x = self._fc2(x)
        return x

    def predict(self, x):
        with torch.no_grad():
            output = self.forward(x).detach()

        return output

    def update(self, q, y):
        self.optimizer.zero_grad()
        loss = self.loss_func(q, y)
        loss.backward()
        loss_value = loss.item()
        self.optimizer.step()
        return loss_value


class Dsqn(nn.Module):
    def __init__(self, history_len, action_num, learning_rate, device, T, tau, v_threshold, v_reset, surrogate_func, alpha):
        super(Dsqn, self).__init__()

        # Define the neural network layers.
        self._static_conv1 = nn.Conv2d(history_len, out_channels=32, kernel_size=8, stride=4)
        self._conv1 = neuron.MultiStepLIFNode(tau, v_threshold, v_reset, getattr(surrogate, surrogate_func)(alpha),
                                              backend="cupy")
        self._conv2 = nn.Sequential(layer.SeqToANNContainer(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4,
                                                                      stride=2)),
                                    neuron.MultiStepLIFNode(tau, v_threshold, v_reset,
                                                            getattr(surrogate, surrogate_func)(alpha), backend="cupy"))
        self._conv3 = nn.Sequential(layer.SeqToANNContainer(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                                                                      stride=1)),
                                    neuron.MultiStepLIFNode(tau, v_threshold, v_reset,
                                                            getattr(surrogate, surrogate_func)(alpha), backend="cupy"))
        self._fc1 = nn.Sequential(layer.SeqToANNContainer(nn.Flatten(),
                                                          nn.Linear(in_features=64 * 7 * 7, out_features=512)),
                                  neuron.MultiStepLIFNode(tau, v_threshold, v_reset, getattr(surrogate, surrogate_func)(alpha),
                                                          backend="cupy"))
        self._fc2 = nn.Linear(in_features=512, out_features=action_num)

        self.device = device
        self._T = T
        self.optimizer = optim.Adam(self.parameters(), learning_rate)
        self.loss_func = nn.SmoothL1Loss()
        self.to(self.device)

    def forward(self, x):
        x_seq = self._static_conv1(x).unsqueeze(dim=0).repeat(self._T, 1, 1, 1, 1)
        x_seq = self._conv1(x_seq)
        x_seq = self._conv2(x_seq)
        x_seq = self._conv3(x_seq)
        x_seq = self._fc1(x_seq)
        x = x_seq.mean(dim=0)
        x = self._fc2(x)
        return x

    def predict(self, x):
        with torch.no_grad():
            output = self.forward(x).detach()

        functional.reset_net(self)
        return output

    def update(self, q, y):
        self.optimizer.zero_grad()
        loss = self.loss_func(q, y)
        loss.backward()
        loss_value = loss.item()
        self.optimizer.step()
        functional.reset_net(self)
        return loss_value


class DsqnSpike(nn.Module):
    def __init__(self, history_len, action_num, learning_rate, device, T, tau, v_threshold, v_reset, surrogate_func, alpha):
        super(DsqnSpike, self).__init__()

        # Define the neural network layers.
        self._static_conv1 = nn.Conv2d(history_len, out_channels=32, kernel_size=8, stride=4)
        self._conv1 = neuron.MultiStepLIFNode(tau, v_threshold, v_reset, getattr(surrogate, surrogate_func)(alpha),
                                              backend="cupy")
        self._conv2 = nn.Sequential(layer.SeqToANNContainer(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4,
                                                                      stride=2)),
                                    neuron.MultiStepLIFNode(tau, v_threshold, v_reset,
                                                            getattr(surrogate, surrogate_func)(alpha), backend="cupy"))
        self._conv3 = nn.Sequential(layer.SeqToANNContainer(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                                                                      stride=1)),
                                    neuron.MultiStepLIFNode(tau, v_threshold, v_reset,
                                                            getattr(surrogate, surrogate_func)(alpha), backend="cupy"))
        self._fc1 = nn.Sequential(layer.SeqToANNContainer(nn.Flatten(),
                                                          nn.Linear(in_features=64 * 7 * 7, out_features=512)),
                                  neuron.MultiStepLIFNode(tau, v_threshold, v_reset, getattr(surrogate, surrogate_func)(alpha),
                                                          backend="cupy"))
        self._fc2 = nn.Sequential(layer.SeqToANNContainer(nn.Linear(in_features=512, out_features=action_num)),
                                  neuron.MultiStepLIFNode(tau, v_threshold, v_reset, getattr(surrogate, surrogate_func)(alpha),
                                                          backend="cupy"))

        self.device = device
        self._T = T
        self.optimizer = optim.Adam(self.parameters(), learning_rate)
        self.loss_func = nn.SmoothL1Loss()
        self.to(self.device)

    def forward(self, x):
        x_seq = self._static_conv1(x).unsqueeze(dim=0).repeat(self._T, 1, 1, 1, 1)
        x_seq = self._conv1(x_seq)
        x_seq = self._conv2(x_seq)
        x_seq = self._conv3(x_seq)
        x_seq = self._fc1(x_seq)
        x_seq = self._fc2(x_seq)
        spike_rate = x_seq.mean(dim=0)
        return spike_rate

    def predict(self, x):
        with torch.no_grad():
            output = self.forward(x).detach()

        functional.reset_net(self)
        return output

    def update(self, q, y):
        self.optimizer.zero_grad()
        loss = self.loss_func(q, y)
        loss.backward()
        loss_value = loss.item()
        self.optimizer.step()
        functional.reset_net(self)
        return loss_value


class DsqnIF(nn.Module):
    def __init__(self, history_len, action_num, learning_rate, device, T, v_threshold, v_reset, surrogate_func, alpha):
        super(DsqnIF, self).__init__()
        
        # Define the neural network layers.
        self._static_conv1 = nn.Conv2d(history_len, out_channels=32, kernel_size=8, stride=4)
        self._conv1 = neuron.MultiStepIFNode(v_threshold, v_reset, getattr(surrogate, surrogate_func)(alpha), backend="cupy")
        self._conv2 = nn.Sequential(layer.SeqToANNContainer(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4,
                                                                      stride=2)),
                                    neuron.MultiStepIFNode(v_threshold, v_reset, getattr(surrogate, surrogate_func)(alpha),
                                                           backend="cupy"))
        self._conv3 = nn.Sequential(layer.SeqToANNContainer(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                                                                      stride=1)),
                                    neuron.MultiStepIFNode(v_threshold, v_reset, getattr(surrogate, surrogate_func)(alpha),
                                                           backend="cupy"))
        self._fc1 = nn.Sequential(layer.SeqToANNContainer(nn.Flatten(),
                                                          nn.Linear(in_features=64 * 7 * 7, out_features=512)),
                                  neuron.MultiStepIFNode(v_threshold, v_reset, getattr(surrogate, surrogate_func)(alpha),
                                                         backend="cupy"))
        self._fc2 = nn.Linear(in_features=512, out_features=action_num)

        self.device = device
        self._T = T
        self.optimizer = optim.Adam(self.parameters(), learning_rate)
        self.loss_func = nn.SmoothL1Loss()
        self.to(self.device)

    def forward(self, x):
        x_seq = self._static_conv1(x).unsqueeze(dim=0).repeat(self._T, 1, 1, 1, 1)
        x_seq = self._conv1(x_seq)
        x_seq = self._conv2(x_seq)
        x_seq = self._conv3(x_seq)
        x_seq = self._fc1(x_seq)
        x = x_seq.mean(dim=0)
        x = self._fc2(x)
        return x

    def predict(self, x):
        with torch.no_grad():
            output = self.forward(x).detach()

        functional.reset_net(self)
        return output

    def update(self, q, y):
        self.optimizer.zero_grad()
        loss = self.loss_func(q, y)
        loss.backward()
        loss_value = loss.item()
        self.optimizer.step()
        functional.reset_net(self)
        return loss_value
