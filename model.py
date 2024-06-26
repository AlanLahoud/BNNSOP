import math
import torch
import torch.nn as nn

# This code defines 2 ANNs and 2 BNNs
# 1 of each is using for each experiment
# Because we needed a more complex network
# for the Quadratic Programming experiment

class VariationalLayer(nn.Module):
    """
    Class to create BNN Layers
    """
    def __init__(self, 
                 input_size, output_size,
                 prior_mu, prior_rho,
                 n_samples, dev, mu_init_1=-0.2, mu_init_2=0.2, rho_init=-5
                ):
        super().__init__()
        
        self.dev = dev
        
        # Bias weight
        input_size = input_size + 1
        
        # Defining Prior distribution (Gaussian)
        self.prior_mu = torch.tensor(prior_mu).to(dev)
        self.prior_rho = torch.tensor(prior_rho).to(dev)
        
        # Defining Variational class (Gaussian class)
        self.theta_mu = nn.Parameter(
            torch.Tensor(input_size, output_size).to(dev).uniform_(
                mu_init_1, mu_init_2)).float()
        self.theta_rho = nn.Parameter(
            torch.Tensor(input_size, output_size).to(dev).uniform_(
                rho_init, rho_init+1)).float()      
        
        # Defining some constants
        self.logsqrttwopi = torch.log(
            torch.sqrt(2*torch.tensor(math.pi))).to(dev)
        self.K = torch.tensor(1).to(dev)
        
        # Defining number of samples for forward
        self.n_samples = n_samples

    
    def rho_to_sigma(self, rho):
        return torch.log(1 + torch.exp(rho))

    def sample_weight(self):
        w = (self.theta_mu.to(self.dev)
        + self.rho_to_sigma(self.theta_rho.to(self.dev))*torch.randn(
            (self.n_samples, self.theta_mu.shape[0], self.theta_mu.shape[1])
        ).to(self.dev))
        return w

    def log_prob_gaussian(self, x, mu, rho):
            return (
                - self.logsqrttwopi
                - torch.log(self.rho_to_sigma(rho))
                - ((x - mu)**2)/(2*self.rho_to_sigma(rho)**2)
            ).sum(axis=[1, 2]).mean()
    
    def prior(self, w):
        return self.log_prob_gaussian(
            w, self.prior_mu, self.prior_rho)
        
    def variational(self, w):
        return self.log_prob_gaussian(
            w, self.theta_mu, self.theta_rho) 
    
    def kl_divergence_layer(self):
        w = self.sample_weight()
        Q = self.variational(w)
        P = self.prior(w)
        KL = Q - P
        return KL
    
    def forward(self, x_layer):
        w = self.sample_weight().to(self.dev)    
        x_next_layer = torch.bmm(x_layer.to(self.dev), w[:, :-1, :]) + w[:,-1,:].unsqueeze(1)
        return x_next_layer
    
    
class VariationalNet(nn.Module):
    def __init__(self, n_samples, input_size, output_size, plv, dev):
        super().__init__()
        self.output_type_dist = True
        self.n_samples = n_samples
        self.act1 = nn.ReLU()
        # Hidden layer sizes
        hl_sizes = [64, 32] 
        mu_init = 0.2
        rho_init=-5
        self.linear1 = VariationalLayer(input_size, hl_sizes[0], 0, plv, n_samples, dev, -mu_init, mu_init, rho_init)
        self.linear2 = VariationalLayer(hl_sizes[0], hl_sizes[1], 0, plv, n_samples, dev, -mu_init, mu_init, rho_init)
        self.linear3 = VariationalLayer(hl_sizes[1], hl_sizes[1], 0, plv, n_samples, dev, -mu_init, mu_init, rho_init)
        self.linear4 = VariationalLayer(hl_sizes[1], output_size, 0, plv, n_samples, dev, -mu_init, mu_init, rho_init)
        self.linear4_2 = VariationalLayer(hl_sizes[1], output_size, 0, plv, n_samples, dev, -0.0001, 0.0001, rho_init-2)
        self.neurons = (
            (input_size+1)*hl_sizes[0] 
            + (hl_sizes[0]+1)*hl_sizes[1]
            + (hl_sizes[1]+1)*hl_sizes[1]
            + 2*(hl_sizes[1]+1)*output_size
        )

    
    def forward(self, x):
        x = torch.unsqueeze(x, 0)
        x = x.expand((self.n_samples, x.shape[1], x.shape[2]))
        x = self.linear1(x)
        x = self.act1(x)

        x = self.linear2(x)
        x = self.act1(x)

        x = self.linear3(x)
        x = self.act1(x)

        y_avg = self.linear4(x)
        rho = self.linear4_2(x)
        return y_avg, rho
    
    def forward_dist(self, x, aleat_bool):
        # Considering epistemic (if BNN) and aleatoric uncertainty
        if aleat_bool:
            y_dist = torch.normal(self(x)[0], torch.sqrt(torch.exp(self(x)[1])))
        else:
            y_dist = self(x)[0]
            
        return y_dist
    
    def kl_divergence_NN(self):
        kl = (
            self.linear1.kl_divergence_layer() 
            + self.linear2.kl_divergence_layer()
            + self.linear3.kl_divergence_layer()
            + self.linear4.kl_divergence_layer()
            + self.linear4_2.kl_divergence_layer()
        )/self.neurons
        return kl
    
    def update_n_samples(self, n_samples):
        self.n_samples = n_samples
        self.linear1.n_samples = n_samples
        self.linear2.n_samples = n_samples
        self.linear3.n_samples = n_samples
        self.linear4.n_samples = n_samples
        self.linear4_2.n_samples = n_samples
        
        
class StandardNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.n_samples = 1
     
        hl_sizes = [128, 64] 
        self.act1 = nn.ReLU()
        self.linear1 = nn.Linear(input_size, hl_sizes[0])
        self.linear2 = nn.Linear(hl_sizes[0], hl_sizes[1])
        self.linear3 = nn.Linear(hl_sizes[1], hl_sizes[1])
        self.linear4 = nn.Linear(hl_sizes[1], output_size)
        self.linear4_2 = nn.Linear(hl_sizes[1], output_size)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act1(x)
        x = self.linear3(x)
        x = self.act1(x)
        y_avg = self.linear4(x)
        rho = self.linear4_2(x)
        return y_avg, rho  
    
    def update_n_samples(self, n_samples):
        self.n_samples = n_samples
        
    def forward_dist(self, x, aleat_bool):
        y, rho = self(x)
        y = y.unsqueeze(0)
        y = y.expand(self.n_samples, -1, -1).clone()
        rho = rho.unsqueeze(0)
        rho = rho.expand(self.n_samples, -1, -1).clone()
        
        # Considering aleatoric uncertainty
        if aleat_bool:
            y_dist = torch.normal(y, torch.sqrt(torch.exp(rho)))
        else:
            y_dist = y

        return y_dist
    
    
class StrongStandardNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.n_samples = 1
     
        hl_sizes = [512, 128] 
        self.act1 = nn.ReLU()
        self.linear1 = nn.Linear(input_size, hl_sizes[0])
        self.linear2 = nn.Linear(hl_sizes[0], hl_sizes[1])
        self.linear3 = nn.Linear(hl_sizes[1], hl_sizes[1])
        self.linear4 = nn.Linear(hl_sizes[1], output_size)
        self.linear4_2 = nn.Linear(hl_sizes[1], output_size)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act1(x)
        x = self.linear3(x)
        x = self.act1(x)
        y_avg = self.linear4(x)
        rho = self.linear4_2(x)
        return y_avg, rho   
    
    def update_n_samples(self, n_samples):
        self.n_samples = n_samples
        
    def forward_dist(self, x, aleat_bool):
        y, rho = self(x)
        y = y.unsqueeze(0)
        y = y.expand(self.n_samples, -1, -1).clone()
        rho = rho.unsqueeze(0)
        rho = rho.expand(self.n_samples, -1, -1).clone()
        
        # Considering aleatoric uncertainty
        if aleat_bool:
            y_dist = torch.normal(y, torch.sqrt(torch.exp(rho)))
        else:
            y_dist = y

        return y_dist
          

class StrongVariationalNet(nn.Module):
    def __init__(self, n_samples, input_size, output_size, plv, dev, var=-0.0001):
        super().__init__()
        self.output_type_dist = True
        self.n_samples = n_samples
        self.act1 = nn.ReLU()
        # Hidden layer sizes
        hl_sizes = [512, 128] 
        mu_init=0.1
        rho_init=-5
        self.linear1 = VariationalLayer(input_size, hl_sizes[0], 0, plv, n_samples, dev, -mu_init, mu_init, rho_init)
        self.linear2 = VariationalLayer(hl_sizes[0], hl_sizes[1], 0, plv, n_samples, dev, -mu_init, mu_init, rho_init)
        self.linear3 = VariationalLayer(hl_sizes[1], hl_sizes[1], 0, plv, n_samples, dev, -mu_init, mu_init, rho_init)
        self.linear4 = VariationalLayer(hl_sizes[1], output_size, 0, plv, n_samples, dev, -mu_init, mu_init, rho_init)
        self.linear4_2 = VariationalLayer(hl_sizes[1], output_size, 0, plv, n_samples, dev, var, var+0.0002, rho_init-2)
        self.neurons = (
            (input_size+1)*hl_sizes[0] 
            + (hl_sizes[0]+1)*hl_sizes[1]
            + (hl_sizes[1]+1)*hl_sizes[1]
            + 2*(hl_sizes[1]+1)*output_size
        )

    
    def forward(self, x):
        x = torch.unsqueeze(x, 0)
        x = x.expand((self.n_samples, x.shape[1], x.shape[2]))
        x = self.linear1(x)
        x = self.act1(x)

        x = self.linear2(x)
        x = self.act1(x)

        x = self.linear3(x)
        x = self.act1(x)

        y_avg = self.linear4(x)
        rho = self.linear4_2(x)
        return y_avg, rho
    
    def forward_dist(self, x, aleat_bool):
        # Considering epistemic (if BNN) and aleatoric uncertainty
        if aleat_bool:
            y_dist = torch.normal(self(x)[0], torch.sqrt(torch.exp(self(x)[1])))
        else:
            y_dist = self(x)[0]
            
        return y_dist
    
    def kl_divergence_NN(self):
        kl = (
            self.linear1.kl_divergence_layer() 
            + self.linear2.kl_divergence_layer()
            + self.linear3.kl_divergence_layer()
            + self.linear4.kl_divergence_layer()
            + self.linear4_2.kl_divergence_layer()
        )/self.neurons
        return kl
    
    def update_n_samples(self, n_samples):
        self.n_samples = n_samples
        self.linear1.n_samples = n_samples
        self.linear2.n_samples = n_samples
        self.linear3.n_samples = n_samples
        self.linear4.n_samples = n_samples
        self.linear4_2.n_samples = n_samples 
        
        
        
        
        
class POStandardNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.n_samples = 1
     
        hl_sizes = [512, 128] 
        self.act1 = nn.ReLU()
        self.linear1 = nn.Linear(input_size, hl_sizes[0])
        self.linear2 = nn.Linear(hl_sizes[0], hl_sizes[1])
        self.linear3 = nn.Linear(hl_sizes[1], hl_sizes[1])
        self.linear4 = nn.Linear(hl_sizes[1], output_size)
        self.linear4_2 = nn.Linear(hl_sizes[1], output_size)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act1(x)
        x = self.linear3(x)
        x = self.act1(x)
        y_avg = self.linear4(x)
        rho = self.act1(self.linear4_2(x)) - 2
        return y_avg, rho   
    
    def update_n_samples(self, n_samples):
        self.n_samples = n_samples
        
    def forward_dist(self, x, aleat_bool):
        y, rho = self(x)
        y = y.unsqueeze(0)
        y = y.expand(self.n_samples, -1, -1).clone()
        rho = rho.unsqueeze(0)
        rho = rho.expand(self.n_samples, -1, -1).clone()
        
        # Considering aleatoric uncertainty
        if aleat_bool:
            y_dist = torch.normal(y, torch.sqrt(torch.exp(rho)))
        else:
            y_dist = y

        return y_dist
          

class POVariationalNet(nn.Module):
    def __init__(self, n_samples, input_size, output_size, plv, dev, var=-0.0001):
        super().__init__()
        self.output_type_dist = True
        self.n_samples = n_samples
        self.act1 = nn.ReLU()
        # Hidden layer sizes
        hl_sizes = [512, 128] 
        mu_init=0.1
        rho_init=-5
        self.linear1 = VariationalLayer(input_size, hl_sizes[0], 0, plv, n_samples, dev, -mu_init, mu_init, rho_init)
        self.linear2 = VariationalLayer(hl_sizes[0], hl_sizes[1], 0, plv, n_samples, dev, -mu_init, mu_init, rho_init)
        self.linear3 = VariationalLayer(hl_sizes[1], hl_sizes[1], 0, plv, n_samples, dev, -mu_init, mu_init, rho_init)
        self.linear4 = VariationalLayer(hl_sizes[1], output_size, 0, plv, n_samples, dev, -mu_init, mu_init, rho_init)
        self.linear4_2 = VariationalLayer(hl_sizes[1], output_size, 0, plv, n_samples, dev, var, var+0.0002, rho_init-2)
        self.neurons = (
            (input_size+1)*hl_sizes[0] 
            + (hl_sizes[0]+1)*hl_sizes[1]
            + (hl_sizes[1]+1)*hl_sizes[1]
            + 2*(hl_sizes[1]+1)*output_size
        )

    
    def forward(self, x):
        x = torch.unsqueeze(x, 0)
        x = x.expand((self.n_samples, x.shape[1], x.shape[2]))
        x = self.linear1(x)
        x = self.act1(x)

        x = self.linear2(x)
        x = self.act1(x)

        x = self.linear3(x)
        x = self.act1(x)

        y_avg = self.act1(self.linear4(x) + 1000) - 1000
        y_avg = -self.act1(-y_avg + 1000) + 1000
        
        rho = self.act1(self.linear4_2(x) + 2) - 2
        rho = -self.act1(-rho + 2) + 2
        return y_avg, rho
    
    def forward_dist(self, x, aleat_bool):
        # Considering epistemic (if BNN) and aleatoric uncertainty
        if aleat_bool:
            y_dist = torch.normal(self(x)[0], torch.sqrt(torch.exp(self(x)[1])))
        else:
            y_dist = self(x)[0]
            
        return y_dist
    
    def kl_divergence_NN(self):
        kl = (
            self.linear1.kl_divergence_layer() 
            + self.linear2.kl_divergence_layer()
            + self.linear3.kl_divergence_layer()
            + self.linear4.kl_divergence_layer()
            + self.linear4_2.kl_divergence_layer()
        )/self.neurons
        return kl
    
    def update_n_samples(self, n_samples):
        self.n_samples = n_samples
        self.linear1.n_samples = n_samples
        self.linear2.n_samples = n_samples
        self.linear3.n_samples = n_samples
        self.linear4.n_samples = n_samples
        self.linear4_2.n_samples = n_samples 
        

        