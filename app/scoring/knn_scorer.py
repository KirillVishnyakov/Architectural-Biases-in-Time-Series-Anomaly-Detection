import torch
import app.utils.config as config
from sklearn.decomposition import PCA

def nearest_neighbor_averaged_distance(R_test, R_train, R_train_norm, k=7):
    """ mean k-nearest squared residual distance to test tensor 
    (euclidean distance but not taking the root)

    For each test residual tensor, compute its distance to all training residuals
    and return the mean distance from the k nearest tensors.

    Args
    ---------
    R_test : tensor (B, D) -> (batch, flattened sample residual)
        normalized test/val residual sample to compare with the train residuals
    R_train : tensor (N, D) -> (all samples, flattened sample residua)
        normalized entire train residuals
    R_train_norm : tensor (1, N)
        unsqueezed precomputed R_train l2 norm
    k : int
        how many nearest neighbors to find
    
    Returns
    ---------
    score : tensor (B, )
        residual score for each sample in batch, 
        defined as the mean of the k nearest neighbors to R_test in R_train

    Example
    ---------
    >>> residual = torch.randn(2, 100)
    >>> train_residuals = torch.randn(32, 100)
    >>> R_train_norm = (train_residuals ** 2).sum(dim=1).unsqueeze(0)
    >>> distance = nearest_neighbor_averaged_distance(residual, train_residuals, R_train_norm)
    >>> distance.shape
    torch.Size([B])
    """
    B, D = R_test.shape 
    N = R_train_norm.shape[1]

    # broadcast explicitly for clarity
    R_test_norm = (R_test ** 2).sum(dim=1, keepdim = True).expand(B, N) # [B, 1] -> [B, N]
    R_train_norm = R_train_norm.expand(B, N) # [1, N] -> [B, N]

    # ||R_test - R_Train||^2 = ||R_test||^2 + ||R_Train||^2 - 2 * R_test @ R_Train.T
    dists = R_train_norm + R_test_norm - 2 * R_test @ R_train.T
    # matrix where each element is the distance of the ith val residual to the jth train residual 

    # get k nearest neighbors, 
    # select the k smallest distances and create a matrix [B, k]
    knn_dists, _ = torch.topk(dists, k, largest=False)

    # average distance, mean knn distances for each sample in the batch
    return knn_dists.mean(dim=1)

class KNNResidualScorer:
    def __init__(self, device, n_components=0.95, k=4, chunk_size = 6000):
        self.device = device
        self.n_components = n_components
        self.k = k
        self.chunk_size = chunk_size

    def name(self):
        return f"knn_pca={self.n_components}_k={self.k}"
    
    def fit(self, train_residuals):
        train_mean = train_residuals.mean(dim=0)
        train_std = train_residuals.std(dim=0) + 1e-8
        train_residuals_normalized = (train_residuals - train_mean) / train_std

        pca = PCA(n_components=self.n_components)
        train_np = train_residuals_normalized.cpu().numpy()
        train_np = pca.fit_transform(train_np)
        train_residuals_normalized = torch.tensor(train_np, dtype=torch.float32)

        train_residuals_normalized = train_residuals_normalized.to(device=self.device, dtype=torch.float32)

        R_train_norm = (train_residuals_normalized ** 2).sum(dim=1).unsqueeze(0)

        self.train_residuals_n = train_residuals_normalized
        self.train_mean = train_mean
        self.train_std = train_std
        self.R_train_norm = R_train_norm
        self.pca = pca
        return self
    
    def score(self, val_residuals):
        val_residuals = val_residuals.to(self.device)

        residuals = (val_residuals - self.train_mean.to(self.device)) / self.train_std.to(self.device)

        residuals_np = residuals.cpu().numpy()
        residuals_np = self.pca.transform(residuals_np)
        residuals = torch.tensor(residuals_np, device=self.device, dtype=torch.float32)

        scores = torch.zeros(len(residuals), device=self.device, dtype=torch.float32)

        for chunk in range(0, len(residuals), self.chunk_size):
            scores[chunk: chunk + self.chunk_size] = nearest_neighbor_averaged_distance(
                residuals[chunk: chunk + self.chunk_size],
                self.train_residuals_n,
                self.R_train_norm,
                self.k
            ) # [N, ]
        return scores 

