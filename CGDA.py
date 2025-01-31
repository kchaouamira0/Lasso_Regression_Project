import numpy as np

#seuillage
def soft_thresholding(x, theta):
    return np.sign(x) * np.maximum(np.abs(x) - theta, 0)

def cgda(X, y, lambda_, num_iterations):
  
    n, p = X.shape
    beta = np.zeros(p)  


    XTX = X.T @ X
    XTy = X.T @ y

    for _ in range(num_iterations):
        for j in range(p):
            # mise à jour  de la coordonnée j
            residual = XTy[j] - np.sum(XTX[j, :] * beta) + XTX[j, j] * beta[j]
            beta[j] = soft_thresholding(residual, n * lambda_) / XTX[j, j]

    return beta

# Exemple 
if __name__ == "__main__":
    np.random.seed(42)
    n, p = 100, 20
    X = np.random.randn(n, p)
    beta_vrai = np.random.randn(p)
    y = X @ beta_vrai + 0.1 * np.random.randn(n)  

    lambda_ = 0.1
    num_iterations = 1000

    beta_estime = cgda(X, y, lambda_, num_iterations)
    print("Coefficients estimés par CGDA :")
    print(beta_estime)
