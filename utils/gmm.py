import torch
from torch import nn
from torch.distributions import MultivariateNormal
from typing import Optional, Tuple
import importlib
import utils.metrics
importlib.reload(utils.metrics)
from utils.metrics import ClusteringMetrics


#####################################################################################
#  Gaussian Mixture Model (GMM) using the Expectation-Maximization (EM) algorithm.  #
#####################################################################################


class GaussianMixture(nn.Module):
    """
    Gaussian Mixture Model (GMM) using the Expectation-Maximization (EM) algorithm.

    Parameters
    ----------
    n_features : int
        The number of features in the dataset.
    n_components : int, default=1
        The number of mixture components.
    covariance_type : {'full', 'tied', 'diag', 'spherical'}, default='full'
        The type of covariance parameters to use.
    tol : float, default=1e-4
        Convergence threshold.
    reg_covar : float, default=1e-6
        Non-negative regularization added to the diagonal of covariance.
    max_iter : int, default=1000
        The number of EM iterations to perform.
    init_params : {'random', 'points', 'kpp', 'kmeans', 'maxdist'}, default='random'
        The method used to initialise the weights, the means, and the covariances.
    weights_init : tensor, default=None
        The initial weights.
    means_init : tensor, default=None
        The initial means.
    covariances_init : tensor, default=None
        The initial covariances.
    random_state : int, default=None
        The seed used by the random number generator.
    warm_start : bool, default=False
        If True, reuse the solution of the last fit.
    verbose : bool, default=False
        Enable verbose output.
    verbose_interval : int, default=10
        Number of iteration done before the next print.
    device : {'cpu', 'cuda'}, default=None
        The device on which the model is run.

    Prior Parameters (Optional)
    ---------------------------
    weight_concentration_prior : Optional[torch.Tensor], default=None
        The Dirichlet concentration parameters for the mixture weights.
        If None, priors on weights are not used (MLE framework).
    mean_prior : Optional[torch.Tensor], default=None
        The prior means for the Gaussian components.
        If None, priors on means are not used (MLE framework).
    mean_precision_prior : Optional[float], default=None
        The precision (inverse variance) for the Gaussian mean priors.
        If None, priors on means are not used (MLE framework).
    degrees_of_freedom_prior : Optional[float], default=None
        The degrees of freedom for the Inverse Wishart covariance priors.
        If None, priors on covariances are not used (MLE framework).
    covariance_prior : Optional[torch.Tensor], default=None
        The scale matrix for the Inverse Wishart covariance priors.
        If None, priors on covariances are not used (MLE framework).

    Attributes
    ----------
    weights_ : tensor
        The weights of each mixture component.
    means_ : tensor
        The mean of each mixture component.
    covariances_ : tensor
        The covariance of each mixture component.
    fitted_ : bool
        True if the model is fitted, False otherwise.
    converged_ : bool
        True if the model has converged, False otherwise.
    n_iter_ : int
        Number of iterations used by the best fit.
    lower_bound_ : float
        Log-likelihood of the best fit.
    """

    def __init__(
            self,
            n_features: int,
            n_components: int = 1,
            covariance_type: str = 'full',
            tol: float = 1e-4,
            reg_covar: float = 1e-6,
            max_iter: int = 1000,
            init_params: str = 'random',
            weights_init: Optional[torch.Tensor] = None,
            means_init: Optional[torch.Tensor] = None,
            covariances_init: Optional[torch.Tensor] = None,
            random_state: Optional[int] = None,
            warm_start: bool = False,
            verbose: bool = False,
            verbose_interval: int = 10,
            device: Optional[str] = None,
            weight_concentration_prior: Optional[torch.Tensor] = None,
            mean_prior: Optional[torch.Tensor] = None,
            mean_precision_prior: Optional[float] = None,
            degrees_of_freedom_prior: Optional[float] = None,
            covariance_prior: Optional[torch.Tensor] = None
        ):
        
        super(GaussianMixture, self).__init__()

        self.n_features = n_features
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.init_params = init_params
        self.weights_init = weights_init
        self.means_init = means_init
        self.covariances_init = covariances_init
        self.random_state = random_state
        self.warm_start = warm_start
        self.verbose = verbose
        self.verbose_interval = verbose_interval
        self.device = torch.device(device if device else "cuda" if torch.cuda.is_available() else "cpu")

        # Initialize prior parameters
        # Flags to indicate whether each prior is active
        self.use_weight_prior = weight_concentration_prior is not None
        self.use_mean_prior = mean_prior is not None and mean_precision_prior is not None
        self.use_covariance_prior = covariance_prior is not None and degrees_of_freedom_prior is not None

        # Set priors only if they are specified
        if self.use_weight_prior:
            if weight_concentration_prior.shape != (self.n_components,):
                raise ValueError(f"weight_concentration_prior must be of shape ({self.n_components},)")
            self.weight_concentration_prior = weight_concentration_prior.to(self.device)
        else:
            self.weight_concentration_prior = None

        if self.use_mean_prior:
            if mean_prior.shape == (self.n_features,):
                mean_prior = mean_prior.unsqueeze(0).repeat(self.n_components, 1)
            elif mean_prior.shape != (self.n_components, self.n_features):
                raise ValueError("mean_prior must be of shape (n_components, n_features) or (n_features,)")
            self.mean_prior = mean_prior.to(self.device)
            self.mean_precision_prior = mean_precision_prior
            if self.mean_precision_prior <= 0:
                raise ValueError("mean_precision_prior must be positive.")
        else:
            self.mean_prior = None
            self.mean_precision_prior = None

        if self.use_covariance_prior:
            if self.covariance_type == 'full':
                expected_shape = (self.n_components, self.n_features, self.n_features)
            elif self.covariance_type == 'tied':
                expected_shape = (self.n_features, self.n_features)
            elif self.covariance_type == 'diag':
                expected_shape = (self.n_components, self.n_features)
            elif self.covariance_type == 'spherical':
                expected_shape = (self.n_components,)
            else:
                raise ValueError("Unsupported covariance type")
            
            if self.covariance_type in ['full', 'diag']:
                if covariance_prior.shape != expected_shape:
                    raise ValueError(f"covariance_prior must be of shape {expected_shape} for '{self.covariance_type}' covariance type.")
            elif self.covariance_type == 'tied':
                if covariance_prior.shape != expected_shape:
                    raise ValueError(f"covariance_prior must be of shape {expected_shape} for '{self.covariance_type}' covariance type.")
            elif self.covariance_type == 'spherical':
                if covariance_prior.dim() != 0:
                    raise ValueError(f"covariance_prior must be a scalar for 'spherical' covariance type.")
            
            self.degrees_of_freedom_prior = degrees_of_freedom_prior
            if self.degrees_of_freedom_prior <= self.n_features - 1:
                raise ValueError(f"degrees_of_freedom_prior must be greater than {self.n_features - 1}, but got {self.degrees_of_freedom_prior}.")
            self.covariance_prior = covariance_prior.to(self.device)
        else:
            self.degrees_of_freedom_prior = None
            self.covariance_prior = None

        # Initialize parameters
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.fitted_ = False
        self.lower_bound_ = None
        self.converged_ = False
        self.n_iter_ = 0

        self._init_params()

    def _init_params(self):
        """
        Initialize model parameters.
        """
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        
        # Initialize weights
        if self.weights_init is not None:
            if self.weights_init.shape != (self.n_components,):
                raise ValueError(f"weights_init must be of shape ({self.n_components},)")
            self.weights_ = self.weights_init.to(self.device)
            if torch.sum(self.weights_) == 0:
                raise ValueError("Initial weights must not sum to zero.")
            self.weights_ = self.weights_ / torch.sum(self.weights_)
        else:
            self.weights_ = torch.full((self.n_components,), 1.0 / self.n_components, dtype=torch.float32, device=self.device)
        
        # Initialize means
        if self.means_init is not None:
            if self.means_init.shape != (self.n_components, self.n_features):
                raise ValueError(f"means_init must be of shape ({self.n_components}, {self.n_features})")
            self.means_ = self.means_init.to(self.device)
        else:
            self.means_ = torch.randn(self.n_components, self.n_features, device=self.device)
        
        # Initialize covariances
        if self.covariances_init is not None:
            if self.covariance_type == 'full':
                expected_shape = (self.n_components, self.n_features, self.n_features)
            elif self.covariance_type == 'tied':
                expected_shape = (self.n_features, self.n_features)
            elif self.covariance_type == 'diag':
                expected_shape = (self.n_components, self.n_features)
            elif self.covariance_type == 'spherical':
                expected_shape = (self.n_components,)
            else:
                raise ValueError("Unsupported covariance type")
            
            if self.covariances_init.shape != expected_shape:
                raise ValueError(f"covariances_init must be of shape {expected_shape} for '{self.covariance_type}' covariance type.")
            self.covariances_ = self.covariances_init.to(self.device)
        else:
            if self.covariance_type == 'full':
                self.covariances_ = torch.stack([
                    torch.eye(self.n_features, device=self.device) * (1.0 + self.reg_covar)
                    for _ in range(self.n_components)
                ])
            elif self.covariance_type == 'diag':
                self.covariances_ = torch.ones(self.n_components, self.n_features, device=self.device) * (1.0 + self.reg_covar)
            elif self.covariance_type == 'spherical':
                self.covariances_ = torch.ones(self.n_components, device=self.device) * (1.0 + self.reg_covar)
            elif self.covariance_type == 'tied':
                self.covariances_ = torch.eye(self.n_features, device=self.device) * (1.0 + self.reg_covar)
            else:
                raise ValueError("Covariance type not supported. Please use 'full', 'tied', 'diag' or 'spherical'.")



    def _e_step(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform the E-step of the EM algorithm.
        """
        X = X.to(self.device)

        log_weights = torch.log(self.weights_ + 1e-10)

        # Compute log_det_cov and Mahalanobis distance
        if self.covariance_type == 'full':
            diff = X.unsqueeze(1) - self.means_.unsqueeze(0)
            try:
                cholesky = torch.linalg.cholesky(self.covariances_)
            except RuntimeError as e:
                raise ValueError(f"Cholesky decomposition failed. Check covariance matrices. Details: {e}")
            log_det_cov = 2.0 * torch.log(torch.diagonal(cholesky, dim1=-2, dim2=-1)).sum(dim=1)
            solve = torch.cholesky_solve(diff.unsqueeze(-1), cholesky)
            mahalanobis = (diff.unsqueeze(-1) * solve).sum(dim=(2, 3))
        elif self.covariance_type == 'tied':
            diff = X.unsqueeze(1) - self.means_.unsqueeze(0)
            try:
                cholesky = torch.linalg.cholesky(self.covariances_)
            except RuntimeError as e:
                raise ValueError(f"Cholesky decomposition failed. Check covariance matrix. Details: {e}")
            log_det_cov = 2.0 * torch.log(torch.diagonal(cholesky)).sum()
            solve = torch.cholesky_solve(diff.unsqueeze(-1), cholesky)
            mahalanobis = (diff.unsqueeze(-1) * solve).sum(dim=(2, 3))
        elif self.covariance_type == 'diag':
            log_det_cov = torch.sum(torch.log(self.covariances_), dim=1)
            precisions = 1.0 / (self.covariances_ + 1e-10)
            diff = X.unsqueeze(1) - self.means_.unsqueeze(0)
            mahalanobis = torch.sum(diff.pow(2) * precisions.unsqueeze(0), dim=2)
        elif self.covariance_type == 'spherical':
            log_det_cov = self.n_features * torch.log(self.covariances_)
            precisions = 1.0 / (self.covariances_ + 1e-10)
            diff = X.unsqueeze(1) - self.means_.unsqueeze(0)
            mahalanobis = torch.sum(diff.pow(2), dim=2) * precisions.unsqueeze(0)
        else:
            raise ValueError("Unsupported covariance type")

        # Compute log probability
        log_prob = -0.5 * (
            self.n_features * torch.log(torch.tensor(2.0 * torch.pi, device=self.device))
            + mahalanobis
            + log_det_cov.unsqueeze(0)
        )
        log_prob += log_weights.unsqueeze(0)

        # Log-sum-exp for normalization
        log_prob_norm = torch.logsumexp(log_prob, dim=1)

        # Responsibilities
        log_resp = log_prob - log_prob_norm.unsqueeze(1)
        resp = torch.exp(log_resp)

        return resp, log_prob_norm

    def _m_step(self, X: torch.Tensor, resp: torch.Tensor):
        """
        Perform the M-step of the EM algorithm with optional MAP updates.
        """
        n_samples, _ = X.shape
        nk = resp.sum(dim=0) + 1e-10

        # Update weights
        if self.use_weight_prior:
            alpha = self.weight_concentration_prior
            total_alpha = alpha.sum()
            self.weights_ = (nk + alpha - 1) / (n_samples + total_alpha - self.n_components)
            self.weights_ = torch.clamp(self.weights_, min=1e-10)
        else:
            self.weights_ = nk / n_samples
            self.weights_ = torch.clamp(self.weights_, min=1e-10)

        # Update means
        if self.use_mean_prior:
            kappa_0 = self.mean_precision_prior
            kappa_k = kappa_0 + nk
            numerator = (resp.t() @ X) + (kappa_0 * self.mean_prior)
            denominator = nk.unsqueeze(1) + kappa_0
            self.means_ = numerator / denominator
        else:
            self.means_ = (resp.t() @ X) / nk.unsqueeze(1)

        # Update covariances
        if self.use_covariance_prior:
            if self.covariance_type == 'full':
                diff = X.unsqueeze(1) - self.means_.unsqueeze(0)
                weighted_diff = resp.unsqueeze(2).unsqueeze(3) * (diff.unsqueeze(3) * diff.unsqueeze(2))
                sum_weighted_diff = weighted_diff.sum(dim=0)

                mean_diff = (self.means_ - self.mean_prior).unsqueeze(-1)
                prior_term_cov = (nk / (nk + self.mean_precision_prior)).unsqueeze(-1).unsqueeze(-1) * torch.matmul(mean_diff, mean_diff.transpose(-1, -2))

                df = self.degrees_of_freedom_prior + nk.unsqueeze(-1).unsqueeze(-1) + self.n_features
                self.covariances_ = (self.covariance_prior + sum_weighted_diff + prior_term_cov + self.reg_covar * torch.eye(self.n_features, device=self.device).unsqueeze(0)) / df

            elif self.covariance_type == 'tied':
                diff = X.unsqueeze(1) - self.means_.unsqueeze(0)
                sum_weighted_diff = torch.einsum('nk,nkd,nke->de', resp, diff, diff)

                mean_diff = (self.means_ - self.mean_prior).unsqueeze(-1)
                prior_term_cov = (nk / (nk + self.mean_precision_prior)).unsqueeze(-1).unsqueeze(-1) * torch.matmul(mean_diff, mean_diff.transpose(-1, -2))
                prior_term_cov = prior_term_cov.sum(dim=0)

                df = self.degrees_of_freedom_prior + nk.sum() + self.n_features
                self.covariances_ = (self.covariance_prior + sum_weighted_diff + prior_term_cov + self.reg_covar * torch.eye(self.n_features, device=self.device)) / df

            elif self.covariance_type == 'diag':
                diff = X.unsqueeze(1) - self.means_.unsqueeze(0)
                weighted_diff = resp.unsqueeze(2) * diff.pow(2)
                sum_weighted_diff = weighted_diff.sum(dim=0)

                mean_diff2 = (self.means_ - self.mean_prior).pow(2)
                prior_term_cov = (nk / (nk + self.mean_precision_prior)).unsqueeze(1) * mean_diff2

                df = self.degrees_of_freedom_prior + nk.unsqueeze(1) + self.n_features
                self.covariances_ = (self.covariance_prior + sum_weighted_diff + prior_term_cov + self.reg_covar) / df

            elif self.covariance_type == 'spherical':
                diff = X.unsqueeze(1) - self.means_.unsqueeze(0)
                diff2 = diff.pow(2).sum(dim=2)
                weighted_diff2 = resp * diff2
                sum_weighted_diff2 = weighted_diff2.sum(dim=0)

                mean_diff2 = (self.means_ - self.mean_prior).pow(2).sum(dim=1)
                prior_term_cov = (nk / (nk + self.mean_precision_prior)) * mean_diff2

                df = self.degrees_of_freedom_prior + nk + self.n_features
                self.covariances_ = (self.covariance_prior + sum_weighted_diff2 + prior_term_cov + self.reg_covar) / (df * self.n_features)
            else:
                raise ValueError("Unsupported covariance type")
        else:
            # MLE Updates without Priors
            if self.covariance_type == 'full':
                diff = X.unsqueeze(1) - self.means_.unsqueeze(0)
                weighted_diff = resp.unsqueeze(2).unsqueeze(3) * (diff.unsqueeze(3) * diff.unsqueeze(2))
                sum_weighted_diff = weighted_diff.sum(dim=0)
                self.covariances_ = sum_weighted_diff / nk.unsqueeze(-1).unsqueeze(-1)
                self.covariances_ += self.reg_covar * torch.eye(self.n_features, device=self.device).unsqueeze(0)
            elif self.covariance_type == 'tied':
                diff = X.unsqueeze(1) - self.means_.unsqueeze(0)
                sum_weighted_diff = torch.einsum('nk,nkd,nke->de', resp, diff, diff)
                self.covariances_ = sum_weighted_diff / nk.sum()
                self.covariances_ += self.reg_covar * torch.eye(self.n_features, device=self.device)
            elif self.covariance_type == 'diag':
                diff = X.unsqueeze(1) - self.means_.unsqueeze(0)
                weighted_diff = resp.unsqueeze(2) * diff.pow(2)
                sum_weighted_diff = weighted_diff.sum(dim=0)
                self.covariances_ = sum_weighted_diff / nk.unsqueeze(1)
                self.covariances_ += self.reg_covar
            elif self.covariance_type == 'spherical':
                diff = X.unsqueeze(1) - self.means_.unsqueeze(0)
                diff2 = diff.pow(2).sum(dim=2)
                weighted_diff2 = resp * diff2
                sum_weighted_diff2 = weighted_diff2.sum(dim=0)
                self.covariances_ = sum_weighted_diff2 / (nk * self.n_features)
                self.covariances_ += self.reg_covar
            else:
                raise ValueError("Unsupported covariance type")



    def _init_random(self, data, k):
        """
        Initialise means randomly based on data distribution.
        """
        mu = torch.mean(data, dim=0)
        if data.dim() == 1:
            cov = torch.var(data)
            samples = torch.randn(k, device=data.device) * torch.sqrt(cov)
        else:
            cov = torch.cov(data.t())
            samples = torch.randn(k, data.size(1), device=data.device) @ torch.linalg.cholesky(cov).t()
        samples += mu
        return samples

    def _init_points(self, data, k):
        """
        Initialise means by randomly selecting points from the data.
        """
        indices = torch.randperm(data.size(0), device=data.device)[:k]
        return data[indices]

    def _init_kpp(self, data, k):
        """
        Initialise means using the k-means++ algorithm.
        """
        n_samples, _ = data.shape
        centroids = torch.empty((k, data.size(1)), device=data.device)
        initial_idx = torch.randint(0, n_samples, (1,), device=data.device)
        centroids[0] = data[initial_idx]

        for i in range(1, k):
            dist_sq = torch.cdist(data, centroids[:i]).pow(2).min(dim=1)[0]
            probabilities = dist_sq / dist_sq.sum()
            selected_idx = torch.multinomial(probabilities, 1)
            centroids[i] = data[selected_idx]

        return centroids

    def _init_kmeans(self, data, k, max_iter=1000, atol=1e-4):
        """
        Initialise means using the k-means algorithm.
        """
        centroids = self._init_kpp(data, k)
        
        for _ in range(max_iter):
            distances = torch.cdist(data, centroids)
            labels = torch.argmin(distances, dim=1)
            
            new_centroids = torch.stack([data[labels == i].mean(dim=0) for i in range(k)])
            
            if torch.allclose(centroids, new_centroids, atol=atol):
                break
            
            centroids = new_centroids
        
        return centroids

    def _init_maxdist(self, data, k):
        """
        Initialise means using a modified k-means++ algorithm
        that maximizes the minimum distance between centroids,
        with reevaluation of the first cluster center.
        """
        n_samples, _ = data.shape
        centroids = torch.empty((k, data.size(1)), device=data.device)
        
        initial_idx = torch.randint(0, n_samples, (1,), device=data.device)
        centroids[0] = data[initial_idx]

        for i in range(1, k):
            dist_sq = torch.cdist(data, centroids[:i]).pow(2)
            min_dist = dist_sq.min(dim=1)[0]
            selected_idx = torch.argmax(min_dist)
            centroids[i] = data[selected_idx]

        dist_sq_to_first = torch.cdist(data, centroids[1:]).pow(2)
        min_dist_to_first = dist_sq_to_first.min(dim=1)[0]
        
        new_first_idx = torch.argmax(min_dist_to_first)
        centroids[0] = data[new_first_idx]

        return centroids

    def fit(self, X: torch.Tensor, max_iter: Optional[int] = None, tol: Optional[float] = None, random_state: Optional[int] = None, warm_start: Optional[bool] = None):
        """
        Fit the Gaussian mixture model to the data.
        """
        if random_state is not None:
            self.random_state = random_state
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        if warm_start is None:
            warm_start = self.warm_start

        if not self.warm_start or not self.fitted_:
            self._init_params()

        if max_iter is None:
            max_iter = self.max_iter

        if tol is None:
            tol = self.tol

        X = X.to(self.device)
        if X.dim() == 1:
            X = X.unsqueeze(1)

        # Initialise means based on the specified method
        if not self.fitted_:
            if self.means_init is not None:
                self.means_ = self.means_init.to(self.device)
            else:
                if self.init_params == 'random':
                    self.means_ = self._init_random(X, self.n_components)
                elif self.init_params == 'points':
                    self.means_ = self._init_points(X, self.n_components)
                elif self.init_params == 'kpp':
                    self.means_ = self._init_kpp(X, self.n_components)
                elif self.init_params == 'kmeans':
                    self.means_ = self._init_kmeans(X, self.n_components)
                elif self.init_params == 'maxdist':
                    self.means_ = self._init_maxdist(X, self.n_components)
                else:
                    raise ValueError("Unsupported initialisation method.")

        for n_iter in range(max_iter):
            self.converged_ = False
            prev_lower_bound = self.lower_bound_ if n_iter > 0 else None

            # E-step
            resp, log_prob_norm = self._e_step(X)

            # Compute total log-likelihood
            self.lower_bound_ = log_prob_norm.mean().item()

            # M-step
            self._m_step(X, resp)

            # Check convergence
            rel_change = abs(self.lower_bound_ - prev_lower_bound) / abs(prev_lower_bound) if prev_lower_bound is not None else 1.0
            if rel_change < tol:
                self.converged_ = True
                if self.verbose > 0:
                    print(f"Converged at iteration {n_iter} with lower bound {self.lower_bound_}")
                break

            if self.verbose > 0 and n_iter % self.verbose_interval == 0:
                print(f"Iteration {n_iter}, lower bound: {self.lower_bound_}")

        self.n_iter_ = n_iter
        self.fitted_ = True

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict the labels for the data samples in X using the trained model.
        """
        resp, _ = self._e_step(X)
        labels = torch.argmax(resp, dim=1)
        return labels

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the components' density for each sample.
        """
        resp, _ = self._e_step(X)
        return resp

    def score_samples(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute the log-likelihood of each sample.
        """
        _, log_prob_norm = self._e_step(X)
        return log_prob_norm

    def score(self, X: torch.Tensor, y=None) -> float:
        """
        Compute the per-sample average log-likelihood of the given data X.
        """
        log_prob = self.score_samples(X)
        return log_prob.mean().item()

    def sample(self, n_samples: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate random samples from the fitted Gaussian distribution.
        """
        if not self.fitted_:
            raise ValueError("The model must be fitted before sampling.")

        indices = torch.multinomial(self.weights_, n_samples, replacement=True)
        means = self.means_[indices]
        
        if self.covariance_type == 'full':
            covariances = self.covariances_[indices]
        elif self.covariance_type == 'tied':
            covariances = self.covariances_.unsqueeze(0).repeat(n_samples, 1, 1)
        elif self.covariance_type == 'diag':
            covariances = torch.diag_embed(self.covariances_[indices])
        elif self.covariance_type == 'spherical':
            covariances = torch.eye(self.n_features, device=self.device).unsqueeze(0).repeat(n_samples, 1, 1) * self.covariances_[indices].view(-1, 1, 1)
        else:
            raise ValueError("Unsupported covariance type")
        
        samples = MultivariateNormal(means, covariance_matrix=covariances).sample()
        return samples, indices

    def evaluate_clustering(self, X, true_labels=None, metrics=None):
        """
        Evaluate clustering metrics against true labels.

        Parameters
        ----------
        X : tensor
            Input data.
        true_labels : tensor, default=None
            True labels for the data.
        metrics : list of str, default=None
            List of metrics to evaluate.

        Returns
        -------
        results : dict
            Dictionary of metric results.
        """
        if metrics is None:
            metrics = [
                # supervised clustering metrics
                "rand_score",
                "adjusted_rand_score",
                "mutual_info_score",
                "normalized_mutual_info_score",
                "adjusted_mutual_info_score",
                "fowlkes_mallows_score",
                "homogeneity_score",
                "completeness_score",
                "v_measure_score",
                "purity_score",
                # classification metrics
                "classification_report",
                "confusion_matrix",
                # unsupervised clustering metrics
                "silhouette_score",
                "davies_bouldin_index",
                "calinski_harabasz_score",
                "dunn_index",
                "bic_score",
                "aic_score",
            ]

        pred_labels = self.predict(X).cpu()
        if true_labels is not None:
            true_labels = true_labels.cpu() if isinstance(true_labels, torch.Tensor) else true_labels

        results = {}
        if true_labels is not None:
            if "rand_score" in metrics:
                results["rand_score"] = ClusteringMetrics.rand_score(true_labels, pred_labels)
            if "adjusted_rand_score" in metrics:
                results["adjusted_rand_score"] = ClusteringMetrics.adjusted_rand_score(true_labels, pred_labels)
            if "mutual_info_score" in metrics:
                results["mutual_info_score"] = ClusteringMetrics.mutual_info_score(true_labels, pred_labels)
            if "adjusted_mutual_info_score" in metrics:
                results["adjusted_mutual_info_score"] = ClusteringMetrics.adjusted_mutual_info_score(true_labels, pred_labels)
            if "normalized_mutual_info_score" in metrics:
                results["normalized_mutual_info_score"] = ClusteringMetrics.normalized_mutual_info_score(true_labels, pred_labels)
            if "fowlkes_mallows_score" in metrics:
                results["fowlkes_mallows_score"] = ClusteringMetrics.fowlkes_mallows_score(true_labels, pred_labels)
            if "homogeneity_score" in metrics:
                results["homogeneity_score"] = ClusteringMetrics.homogeneity_score(true_labels, pred_labels)
            if "completeness_score" in metrics:
                results["completeness_score"] = ClusteringMetrics.completeness_score(true_labels, pred_labels)
            if "v_measure_score" in metrics:
                results["v_measure_score"] = ClusteringMetrics.v_measure_score(true_labels, pred_labels)
            if "purity_score" in metrics:
                results["purity_score"] = ClusteringMetrics.purity_score(true_labels, pred_labels)
            if "classification_report" in metrics:
                results["classification_report"] = ClusteringMetrics.classification_report(true_labels, pred_labels)
            if "confusion_matrix" in metrics:
                results["confusion_matrix"] = ClusteringMetrics.confusion_matrix(true_labels, pred_labels)
        
        if "silhouette_score" in metrics:
            results["silhouette_score"] = ClusteringMetrics.silhouette_score(X, pred_labels, self.n_components)
        if "davies_bouldin_index" in metrics:
            results["davies_bouldin_index"] = ClusteringMetrics.davies_bouldin_index(X, pred_labels, self.n_components)
        if "calinski_harabasz_score" in metrics:
            results["calinski_harabasz_score"] = ClusteringMetrics.calinski_harabasz_score(X, pred_labels, self.n_components)
        if "bic_score" in metrics:
            results["bic_score"] = ClusteringMetrics.bic_score(self.lower_bound_, X, self.n_components, self.covariance_type)
        if "aic_score" in metrics:
            results["aic_score"] = ClusteringMetrics.aic_score(self.lower_bound_, X, self.n_components, self.covariance_type)
        if "dunn_index" in metrics:
            results["dunn_index"] = ClusteringMetrics.dunn_index(X, pred_labels, self.n_components)

        return results

    def set_mean_prior(self, mean_prior: torch.Tensor, mean_precision_prior: float):
        """
        Update the mean prior for the GMM.
        
        Parameters:
        - mean_prior (torch.Tensor): New mean prior with shape (n_components, n_features).
        - mean_precision_prior (float): Precision of the mean prior.
        """
        if mean_prior.shape != (self.n_components, self.n_features):
            raise ValueError(f"mean_prior must be of shape ({self.n_components}, {self.n_features})")
        self.mean_prior = mean_prior.to(self.device)
        self.mean_precision_prior = mean_precision_prior
        self.use_mean_prior = True

    def set_covariance_prior(self, covariance_prior: torch.Tensor, degrees_of_freedom_prior: float):
        """
        Update the covariance prior for the GMM.
        
        Parameters:
        - covariance_prior (torch.Tensor): New covariance prior.
        - degrees_of_freedom_prior (float): Degrees of freedom for the covariance prior.
        """
        # Validate shapes based on covariance_type
        if self.covariance_type == 'full':
            expected_shape = (self.n_components, self.n_features, self.n_features)
        elif self.covariance_type == 'tied':
            expected_shape = (self.n_features, self.n_features)
        elif self.covariance_type == 'diag':
            expected_shape = (self.n_components, self.n_features)
        elif self.covariance_type == 'spherical':
            expected_shape = (self.n_components,)
        else:
            raise ValueError("Unsupported covariance type")

        if self.covariance_type in ['full', 'diag']:
            if covariance_prior.shape != expected_shape:
                raise ValueError(f"covariance_prior must be of shape {expected_shape} for '{self.covariance_type}' covariance type.")
        elif self.covariance_type == 'tied':
            if covariance_prior.shape != expected_shape:
                raise ValueError(f"covariance_prior must be of shape {expected_shape} for '{self.covariance_type}' covariance type.")
        elif self.covariance_type == 'spherical':
            if covariance_prior.shape != expected_shape:
                raise ValueError(f"covariance_prior must be of shape {expected_shape} for 'spherical' covariance type.")
        
        self.covariance_prior = covariance_prior.to(self.device)
        self.degrees_of_freedom_prior = degrees_of_freedom_prior
        self.use_covariance_prior = True