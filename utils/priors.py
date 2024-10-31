import torch


"""
Prior for the weights of the model
"""

class DirichletPrior:
    def __init__(self, concentration, device='cpu'):
        self.concentration = concentration.to(device)
        self.device = device
        self._dist = torch.distributions.Dirichlet(self.concentration)
        
    def log_prob(self, x):
        x = x.to(self.device)
        return self._dist.log_prob(x)
    
"""
Priors for the means of the model
"""
    
class GaussianPrior:
    def __init__(self, dim, mean, stddev, device='cpu'):
        self.device = device
        self._dim = dim
        self._mean = mean.to(device)
        self._stddev = torch.tensor(stddev).to(device)
        self._dist = torch.distributions.Normal(self._mean, self._stddev)

    def log_prob(self, x):
        x = x.to(self.device)
        log_prob = self._dist.log_prob(x)
        return log_prob.sum(dim=-1)
    
class SoftballPrior:
    def __init__(self, dim, radius, a=1, device='cpu'):
        self.dim = dim
        self.radius = torch.tensor(radius, device=device)
        self.a = a
        self.device = device
        self.norm = torch.lgamma(torch.tensor(1 + dim * 0.5, device=device)) - dim * (
            torch.log(self.radius) + 0.5 * torch.log(torch.tensor(torch.pi, device=device))
        )

    def log_prob(self, x):
        x = x.to(self.device)
        return self.norm - torch.log(
            1 + torch.exp(self.a * (torch.norm(x, dim=-1) / self.radius - 1))
        )
    
"""
Priors for the covariance matrix of the model
"""
    
       
class WishartPrior:
    def __init__(self, df, scale, covariance_type, device='cpu'):
        self.device = device
        self.df = df
        self.scale = scale.to(device)
        self.covariance_type = covariance_type
        if covariance_type in ['full', 'tied']:
            self._dist = torch.distributions.Wishart(df, self.scale)
        elif covariance_type == 'diag':
            self._dist = torch.distributions.Gamma(df / 2, self.scale.diag() / 2)
        elif covariance_type == 'spherical':
            self._dist = torch.distributions.Gamma(df / 2, self.scale / 2)

    def log_prob(self, x):
        x = x.to(self.device)
        if self.covariance_type == 'full':
            return self._dist.log_prob(x.to(self.device))
        elif self.covariance_type == 'tied':
            return self._dist.log_prob(x.to(self.device))
        elif self.covariance_type == 'diag':
            return self._dist.log_prob(x.to(self.device)).sum()
        elif self.covariance_type == 'spherical':
            return self._dist.log_prob(x.to(self.device)).sum()
        else:
            raise ValueError("Unsupported covariance matrix shape")


class InverseGammaPrior:
    def __init__(self, alpha, beta, device='cpu'):
        self.device = device
        self.alpha = alpha.to(device)
        self.beta = beta.to(device)
        self._dist = torch.distributions.InverseGamma(self.alpha, self.beta)
        
    def log_prob(self, x):
        x = x.to(self.device)
        if x.dim() == 1:  # Diagonal or spherical
            return self._dist.log_prob(x).sum()
        elif x.dim() == 2:  # Full or tied
            diag_elements = torch.diagonal(x)
            return self._dist.log_prob(diag_elements).sum()
        elif x.dim() == 3:  # Full
            diag_elements = torch.diagonal(x, dim1=-2, dim2=-1)
            return self._dist.log_prob(diag_elements).sum(dim=1).sum()
        else:
            raise ValueError("Unsupported covariance matrix shape")

        
class LogNormalPrior:
    def __init__(self, mean, stddev, device='cpu'):
        self.device = device
        self._dist = torch.distributions.LogNormal(mean.to(device), stddev.to(device))

    def log_prob(self, x):
        x = x.to(self.device)
        if x.dim() == 1:  # Diagonal or spherical
            return self._dist.log_prob(x).sum()
        elif x.dim() == 2:  # Tied
            diag_elements = torch.diagonal(x)
            return self._dist.log_prob(diag_elements).sum()
        elif x.dim() == 3:  # Full
            diag_elements = torch.diagonal(x, dim1=-2, dim2=-1)
            return self._dist.log_prob(diag_elements).sum(dim=1).sum()
        else:
            raise ValueError("Unsupported covariance matrix shape")