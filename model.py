import torch
import torch.nn as nn

class VariationalLayer(nn.Module):
    def __init__(self, 
                 input_size, output_size,
                 prior_mu, prior_rho,
                 n_samples
                ):
        super().__init__()
        
        # Bias weight
        input_size = input_size + 1
        
        # Defining Prior distribution (Gaussian)
        self.prior_mu = torch.tensor(prior_mu)
        self.prior_rho = torch.tensor(prior_rho)
        
        # Defining Variational class (Gaussian class)
        self.theta_mu = nn.Parameter(
            torch.Tensor(input_size, output_size).uniform_(
                -0.5, 0.5)).float()
        self.theta_rho = nn.Parameter(
            torch.Tensor(input_size, output_size).uniform_(
                -5,-4)).float()
        
        # Defining some constants
        self.logsqrttwopi = torch.log(
            torch.sqrt(2*torch.tensor(torch.pi)))
        self.K = torch.tensor(1)
        
        # Defining number of samples for forward
        self.n_samples = n_samples
    
    def rho_to_sigma(self, theta_rho):
        return torch.log(1 + torch.exp(theta_rho))

    def sample_weight(self, theta_mu, theta_rho):
        w = (theta_mu 
        + self.rho_to_sigma(theta_rho)*torch.randn(
            (self.n_samples, theta_mu.shape[0], theta_mu.shape[1])
        ))
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
        
    def variational(self, w, theta_mu, theta_rho):
        return self.log_prob_gaussian(
            w, theta_mu, theta_rho) 
    
    def kl_divergence_layer(self):
        theta_mu = self.theta_mu
        theta_rho = self.theta_rho
        w = self.sample_weight(theta_mu, theta_rho)
        Q = self.variational(w, theta_mu, theta_rho)
        P = self.prior(w)
        KL = Q - P 
        return KL
    
    def forward(self, x_layer):
        theta_mu = self.theta_mu
        theta_rho = self.theta_rho
        w = self.sample_weight(theta_mu, theta_rho)    
        x_next_layer = torch.bmm(x_layer, w[:, :-1, :]) + w[:,-1,:].unsqueeze(1)
        return x_next_layer
    
    
class VariationalNet(nn.Module):
    def __init__(self, n_samples, input_size, output_size, plv):
        super().__init__()
        self.output_type_dist = True
        self.n_samples = n_samples
        self.act1 = nn.ReLU()
        self.linear1 = VariationalLayer(input_size, 512, 0, plv, n_samples)
        self.linear2 = VariationalLayer(512, 128, 0, plv, n_samples)
        self.linear3 = VariationalLayer(128, output_size, 0, plv, n_samples)
    
    def forward(self, x):
        x = torch.unsqueeze(x, 0)
        x = x.expand((self.n_samples, x.shape[1], x.shape[2]))
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act1(x)
        x = self.linear3(x)
        return x
    
    def forward_dist(self, x):
        return self(x)
    
    def kl_divergence_NN(self):
        kl = (
            self.linear1.kl_divergence_layer() 
            + self.linear2.kl_divergence_layer()
            + self.linear3.kl_divergence_layer()
        )
        return kl
    
    def update_n_samples(self, n_samples):
        self.n_samples = n_samples
        self.linear1.n_samples = n_samples
        self.linear2.n_samples = n_samples
        self.linear3.n_samples = n_samples
        
    
class StandardNet(nn.Module):
    def __init__(self, input_size, output_size, eps):
        super().__init__()
        self.n_samples = 1
        self.output_type_dist = False
        self.eps = eps
        if eps > 0:
            self.output_type_dist = True
        self.act1 = nn.ReLU()
        self.linear1 = nn.Linear(input_size, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, output_size)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act1(x)
        x = self.linear3(x)
        return x   
    
    def update_n_samples(self, n_samples):
        self.n_samples = n_samples
        
    def forward_dist(self, x):
        y = self(x)
        y = y.unsqueeze(0)
        y = y.expand(self.n_samples, -1, -1).clone()
        y += self.eps*torch.randn(y.shape)/20
        return y
        

        
        
        
        
        
        
        
        
        
        
        