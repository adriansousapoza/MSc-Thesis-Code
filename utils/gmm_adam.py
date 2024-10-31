import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from typing import Optional, Tuple

import importlib
import utils.metrics
importlib.reload(utils.metrics)
from utils.metrics import ClusteringMetrics


class CovarianceHandlerAdam:
    def __init__(self, covariance_type: str, reg_covar: float, n_components: int, n_features: int, device: torch.device):
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.n_components = n_components
        self.n_features = n_features
        self.device = device

    def compute_log_prob(self, X: torch.Tensor, weights: torch.Tensor, means: torch.Tensor, covariances: torch.Tensor) -> torch.Tensor:
        """
        Compute the log probability of the data under the GMM.

        Parameters
        ----------
        X : tensor
            Input data.
        weights : tensor
            Weights of the mixture components.
        means : tensor
            Means of the mixture components.
        covariances : tensor
            Covariances of the mixture components.

        Returns
        -------
        log_prob_norm : tensor
            Log probability of the data.
        """
        n_samples, n_features = X.shape

        # Ensure positive definiteness of covariances
        if self.covariance_type == 'full':
            covariances = covariances + self.reg_covar * torch.eye(n_features, device=self.device).unsqueeze(0)
        elif self.covariance_type == 'diag':
            covariances = covariances + self.reg_covar
        elif self.covariance_type == 'spherical':
            covariances = covariances + self.reg_covar
        elif self.covariance_type == 'tied':
            covariances = covariances + self.reg_covar * torch.eye(n_features, device=self.device)
        else:
            raise ValueError("Unsupported covariance type")

        log_probs = []
        for k in range(self.n_components):
            if self.covariance_type == 'full':
                # Ensure the covariance matrix is positive definite via Cholesky
                L = torch.tril(covariances[k])
                cov = L @ L.T + self.reg_covar * torch.eye(self.n_features, device=self.device)
            elif self.covariance_type == 'diag':
                cov = torch.diag(F.softplus(covariances[k]) + self.reg_covar)
            elif self.covariance_type == 'spherical':
                cov = torch.eye(self.n_features, device=self.device) * (F.softplus(covariances[k]) + self.reg_covar)
            elif self.covariance_type == 'tied':
                L = torch.tril(covariances)
                cov = L @ L.T + self.reg_covar * torch.eye(self.n_features, device=self.device)
            else:
                raise ValueError("Unsupported covariance type")

            dist = MultivariateNormal(means[k], covariance_matrix=cov)
            log_prob = dist.log_prob(X) + torch.log(weights[k])
            log_probs.append(log_prob)

        log_probs = torch.stack(log_probs, dim=1)
        log_prob_norm = torch.logsumexp(log_probs, dim=1)
        return log_prob_norm

    def sample_gaussians(self, means: torch.Tensor, covariances: torch.Tensor, weights: torch.Tensor, n_samples: int) -> torch.Tensor:
        """
        Sample from the Gaussian distributions considering the weights of each component.
        """
        indices = torch.multinomial(weights, n_samples, replacement=True)
        samples = []
        for i in indices:
            k = i.item()
            mean = means[k]
            if self.covariance_type == 'full':
                cov = covariances[k]
            elif self.covariance_type == 'tied':
                cov = covariances
            elif self.covariance_type == 'diag':
                cov = torch.diag(covariances[k])
            elif self.covariance_type == 'spherical':
                cov = torch.eye(self.n_features, device=self.device) * covariances[k]
            else:
                raise ValueError("Unsupported covariance type")
            dist = MultivariateNormal(mean, covariance_matrix=cov)
            sample = dist.sample()
            samples.append(sample)
        return torch.stack(samples)


class GaussianMixtureAdam(nn.Module):
    """
    Gaussian Mixture Model (GMM) optimized using the Adam optimizer.

    Parameters
    ----------
    n_features : int
        The number of features in the dataset.
    n_components : int, default=1
        The number of mixture components.
    covariance_type : {'full', 'tied', 'diag', 'spherical'}, default='full'
        The type of covariance parameters to use.
    tol : float, default=1e-6
        Convergence threshold.
    reg_covar : float, default=1e-6
        Non-negative regularization added to the diagonal of covariance.
    max_iter : int, default=1000
        The number of iterations to perform.
    init_params : {'random', 'points', 'kpp', 'kmeans', 'maxdist'}, default='random'
        The method used to initialize the weights, the means, and the covariances.
    learning_rate : float, default=1e-3
        Learning rate for the optimizer.
    random_state : int, default=None
        The seed used by the random number generator.
    verbose : bool, default=False
        Enable verbose output.
    verbose_interval : int, default=10
        Number of iteration done before the next print.
    device : {'cpu', 'cuda'}, default=None
        The device on which the model is run.

    Attributes
    ----------
    weights_ : tensor
        The weights of each mixture component.
    means_ : tensor
        The mean of each mixture component.
    covariances_ : tensor
        The covariance of each mixture component.
    converged_ : bool
        True if the model has converged, False otherwise.
    n_iter_ : int
        Number of steps used by the best fit.
    lower_bound_ : float
        Log-likelihood of the best fit.

    Methods
    -------
    fit(X, max_iter=None, tol=None, random_state=None):
        Fit the Gaussian mixture model.
    sample(n_samples=1):
        Generate random samples from the fitted Gaussian distribution.
    predict(X):
        Predict the labels for the data samples in X using the trained model.
    score_samples(X):
        Return the per-sample likelihood of the data under the model.
    evaluate_clustering(X, true_labels=None, metrics=None):
        Evaluate supervised and unsupervised clustering metrics.
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
            learning_rate: float = 1e-3,
            random_state: Optional[int] = None,
            verbose: bool = False,
            verbose_interval: int = 10,
            device: Optional[str] = None
        ):

        super(GaussianMixtureAdam, self).__init__()

        self.n_features = n_features
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.init_params = init_params
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.verbose = verbose
        self.verbose_interval = verbose_interval
        self.device = torch.device(device if device else "cuda" if torch.cuda.is_available() else "cpu")

        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.converged_ = False
        self.n_iter_ = 0
        self.lower_bound_ = None

        self.cov_handler = CovarianceHandlerAdam(covariance_type=self.covariance_type, reg_covar=self.reg_covar, n_components=self.n_components, n_features=self.n_features, device=self.device)

        self._init_params()

    def _init_params(self):
        """
        Initialize model parameters.
        """
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        # Initialize weights
        self.weights_ = torch.full((self.n_components,), float(1.0 / self.n_components), dtype=torch.float32, device=self.device, requires_grad=True)

        # Initialize means
        self.means_ = torch.randn(self.n_components, self.n_features, device=self.device, requires_grad=True)

        # Initialize covariances
        if self.covariance_type == 'full':
            self.covariances_ = torch.stack([torch.eye(self.n_features, device=self.device) for _ in range(self.n_components)], dim=0).requires_grad_(True)
        elif self.covariance_type == 'diag':
            self.covariances_ = torch.ones(self.n_components, self.n_features, device=self.device, requires_grad=True)
        elif self.covariance_type == 'spherical':
            self.covariances_ = torch.ones(self.n_components, device=self.device, requires_grad=True)
        elif self.covariance_type == 'tied':
            self.covariances_ = torch.eye(self.n_features, device=self.device, requires_grad=True)
        else:
            raise ValueError("Covariance type not supported. Please use 'full', 'tied', 'diag' or 'spherical'.")

    def _init_krandom(self, data, k):
        """
        Initialize means randomly based on data distribution.

        Parameters
        ----------
        data : tensor
            Input data.
        k : int
            Number of components.

        Returns
        -------
        samples : tensor
            Initialized means.
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

    def _init_kpoints(self, data, k):
        """
        Initialize means by randomly selecting points from the data.

        Parameters
        ----------
        data : tensor
            Input data.
        k : int
            Number of components.

        Returns
        -------
        samples : tensor
            Initialized means.
        """
        indices = torch.randperm(data.size(0), device=data.device)[:k]
        return data[indices]

    def _init_kpp(self, data, k):
        """
        Initialize means using the k-means++ algorithm.

        Parameters
        ----------
        data : tensor
            Input data.
        k : int
            Number of components.

        Returns
        -------
        centroids : tensor
            Initialized means.
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
        Initialize means using the k-means algorithm.

        Parameters
        ----------
        data : tensor
            Input data.
        k : int
            Number of components.
        max_iter : int, default=1000
            Maximum number of iterations.
        atol : float, default=1e-4
            Convergence threshold.

        Returns
        -------
        centroids : tensor
            Initialized means.
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
        Initialize means using a modified k-means++ algorithm
        that maximizes the minimum distance between centroids,
        with reevaluation of the first cluster center.

        Parameters
        ----------
        data : tensor
            Input data.
        k : int
            Number of components.

        Returns
        -------
        centroids : tensor
            Initialized means.
        """
        n_samples, _ = data.shape
        centroids = torch.empty((k, data.size(1)), device=data.device)

        # Select the first centroid randomly
        initial_idx = torch.randint(0, n_samples, (1,), device=data.device)
        centroids[0] = data[initial_idx]

        # Select remaining centroids by maximizing the minimum distance
        for i in range(1, k):
            dist_sq = torch.cdist(data, centroids[:i]).pow(2)
            min_dist = dist_sq.min(dim=1)[0]
            selected_idx = torch.argmax(min_dist)  # Select the point with the maximum minimum distance
            centroids[i] = data[selected_idx]

        # Reevaluate the first centroid
        dist_sq_to_first = torch.cdist(data, centroids[1:]).pow(2)
        min_dist_to_first = dist_sq_to_first.min(dim=1)[0]

        # If the first centroid is too close, replace it with a better option
        new_first_idx = torch.argmax(min_dist_to_first)
        centroids[0] = data[new_first_idx]

        return centroids

    def fit(self, X: torch.Tensor, max_iter: Optional[int] = None, tol: Optional[float] = None, random_state: Optional[int] = None):
        """
        Fit the Gaussian mixture model to the data.
        """
        if random_state is not None:
            self.random_state = random_state
            torch.manual_seed(self.random_state)

        if max_iter is None:
            max_iter = self.max_iter

        if tol is None:
            tol = self.tol

        X = X.to(self.device)

        # Initialize means based on the specified method
        if self.init_params == 'random':
            self.means_.data = self._init_krandom(X, self.n_components)
        elif self.init_params == 'points':
            self.means_.data = self._init_kpoints(X, self.n_components)
        elif self.init_params == 'kpp':
            self.means_.data = self._init_kpp(X, self.n_components)
        elif self.init_params == 'kmeans':
            self.means_.data = self._init_kmeans(X, self.n_components)
        elif self.init_params == 'maxdist':
            self.means_.data = self._init_maxdist(X, self.n_components)
        else:
            raise ValueError("Unsupported initialization method.")

        params = [self.weights_, self.means_, self.covariances_]
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)

        prev_loss = None

        for n_iter in range(max_iter):
            optimizer.zero_grad()

            log_prob_norm = self.cov_handler.compute_log_prob(X, self.weights_, self.means_, self.covariances_)
            loss = -log_prob_norm.mean()

            loss.backward()
            optimizer.step()

            self.lower_bound_ = -loss.item()

            # Ensure weights sum to 1 and are positive
            with torch.no_grad():
                self.weights_.data.clamp_(min=self.reg_covar)
                self.weights_.data /= self.weights_.data.sum()

                if self.covariance_type == 'full':
                    for k in range(self.n_components):
                        eigvals, eigvecs = torch.linalg.eigh(self.covariances_.data[k])
                        eigvals = torch.clamp(eigvals, min=self.reg_covar)
                        self.covariances_.data[k] = eigvecs @ torch.diag(eigvals) @ eigvecs.T
                elif self.covariance_type in ['diag', 'spherical']:
                    self.covariances_.data.clamp_(min=self.reg_covar)

            # Check convergence
            if prev_loss is not None and abs(prev_loss - loss.item()) / abs(prev_loss) < tol:
                if self.verbose:
                    print(f"Converged at iteration {n_iter}")
                break

            prev_loss = loss.item()

            if self.verbose and n_iter % self.verbose_interval == 0:
                print(f"Iteration {n_iter}, loss: {loss.item()}")

        self.converged_ = n_iter < max_iter - 1
        self.n_iter_ = n_iter

    def sample(self, n_samples: int = 1) -> torch.Tensor:
        """
        Generate random samples from the fitted Gaussian distribution.
        """
        samples = self.cov_handler.sample_gaussians(self.means_, self.covariances_, self.weights_, n_samples)
        return samples

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict the labels for the data samples in X using the trained model.
        """
        n_samples, _ = X.shape
        X = X.to(self.device)

        log_probs = []
        for k in range(self.n_components):
            if self.covariance_type == 'full':
                cov = self.covariances_[k] + self.reg_covar * torch.eye(self.n_features, device=self.device)
            elif self.covariance_type == 'tied':
                cov = self.covariances_ + self.reg_covar * torch.eye(self.n_features, device=self.device)
            elif self.covariance_type == 'diag':
                cov = torch.diag(self.covariances_[k] + self.reg_covar)
            elif self.covariance_type == 'spherical':
                cov = torch.eye(self.n_features, device=self.device) * (self.covariances_[k] + self.reg_covar)
            else:
                raise ValueError("Unsupported covariance type")

            dist = MultivariateNormal(self.means_[k], covariance_matrix=cov)
            log_prob = dist.log_prob(X) + torch.log(self.weights_[k])
            log_probs.append(log_prob)

        log_probs = torch.stack(log_probs, dim=1)
        labels = torch.argmax(log_probs, dim=1)
        return labels.cpu()

    def score_samples(self, X: torch.Tensor) -> torch.Tensor:
        """
        Return the per-sample log-likelihood of the data under the model.
        """
        X = X.to(self.device)
        log_prob_norm = self.cov_handler.compute_log_prob(X, self.weights_, self.means_, self.covariances_)
        return log_prob_norm.cpu()

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

        pred_labels = self.predict(X)
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
            results["silhouette_score"] = ClusteringMetrics.silhouette_score(X.cpu(), pred_labels, self.n_components)
        if "davies_bouldin_index" in metrics:
            results["davies_bouldin_index"] = ClusteringMetrics.davies_bouldin_index(X.cpu(), pred_labels, self.n_components)
        if "calinski_harabasz_score" in metrics:
            results["calinski_harabasz_score"] = ClusteringMetrics.calinski_harabasz_score(X.cpu(), pred_labels, self.n_components)
        if "bic_score" in metrics:
            results["bic_score"] = ClusteringMetrics.bic_score(self.lower_bound_, X.cpu(), self.n_components, self.covariance_type)
        if "aic_score" in metrics:
            results["aic_score"] = ClusteringMetrics.aic_score(self.lower_bound_, X.cpu(), self.n_components, self.covariance_type)
        if "dunn_index" in metrics:
            results["dunn_index"] = ClusteringMetrics.dunn_index(X.cpu(), pred_labels, self.n_components)

        return results
