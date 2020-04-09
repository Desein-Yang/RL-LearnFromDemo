import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CNNModel(nn.Module):
    """
    TODO:Network module for PPO.
    """
    def __init__(self, input_dim,action_dim,is_actor):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(kernel_size = 8,stride=4)
        
        self.is_actor = is_actor
        self.set_parameter_no_grad()
        self._ortho_ini()
        # self.previous_frame should be PILImage
        self.previous_frame = None

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        if self.is_actor:
            return F.softmax(x)
        else:
            return x

    def _ortho_ini(self):
        for m in self.modules():
            # Orthogonal initialization and layer scaling
            # Paper name : Implementation Matters in Deep Policy Gradient: A case study on PPO and TRPO
            if isinstance(m,(nn.Linear,nn.Conv2d)):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def set_parameter_no_grad(self):
        for param in self.parameters():
            param.requires_grad = False

    def get_size(self):
        """
        Returns:   
            Number of all params
        """
        count = 0
        for params in self.parameters():
            count += params.numel()
        return count

class MLPModel(nn.Module):
    """
    Network module for PPO.
    """
    def __init__(self, input_dim,action_dim,is_actor):
        super(MLPModel, self).__init__()
        n_latent_var = 64
        self.fc1 = nn.Linear(input_dim, n_latent_var)
        self.fc2 = nn.Linear(n_latent_var, n_latent_var)
        self.fc3 = nn.Linear(n_latent_var, action_dim)
        
        self.is_actor = is_actor
        self.set_parameter_no_grad()
        self._ortho_ini()
        # self.previous_frame should be PILImage
        self.previous_frame = None

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        if self.is_actor:
            return F.softmax(x)
        else:
            return x

    def _ortho_ini(self):
        for m in self.modules():
            # Orthogonal initialization and layer scaling
            # Paper name : Implementation Matters in Deep Policy Gradient: A case study on PPO and TRPO
            if isinstance(m,(nn.Linear,nn.Conv2d)):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def set_parameter_no_grad(self):
        for param in self.parameters():
            param.requires_grad = False

    def get_size(self):
        """
        Returns:   
            Number of all params
        """
        count = 0
        for params in self.parameters():
            count += params.numel()
        return count

