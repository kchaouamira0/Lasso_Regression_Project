import numpy as np

#seuillage
def soft_thresholding(x, theta):
    return np.sign(x) * np.maximum(np.abs(x) - theta, 0)

def ista(X, y, lambda_, L, num_iterations):
  
    n, p = X.shape
    beta = np.zeros(p)

    for _ in range(num_iterations):
        gradient = - (1 / n) * X.T @ (y - X @ beta)
        beta = soft_thresholding(beta - (1 / L) * gradient, lambda_ / L)

    return beta

# Exemple 
if __name__ == "__main__":
    np.random.seed(42)
    n, p = 100, 20
    X = np.random.randn(n, p)
    beta_vrai = np.random.randn(p)
    y = X @ beta_vrai + 0.1 * np.random.randn(n)  # bruit gaussien

    lambda_ = 0.1
    L = np.linalg.norm(X.T @ X, 2) / n  # Approximation de la constante de Lipschitz
    num_iterations = 1000

    beta_estime = ista(X, y, lambda_, L, num_iterations)
    print("Coefficients estimés par ISTA :")
    print(beta_estime)
