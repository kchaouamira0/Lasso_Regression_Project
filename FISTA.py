import numpy as np

#seuillage
def soft_thresholding(x, theta):
    return np.sign(x) * np.maximum(np.abs(x) - theta, 0)

def fista(X, y, lambda_, L, num_iterations):
   
    n, p = X.shape
    beta = np.zeros(p)
    alpha = np.copy(beta)  
    t = 1  

    for _ in range(num_iterations):
       
        gradient = - (1 / n) * X.T @ (y - X @ alpha) 
        beta_new = soft_thresholding(alpha - (1 / L) * gradient, lambda_ / L) #mise a jour de beta
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2 #mise a jour de t
        alpha = beta_new + ((t - 1) / t_new) * (beta_new - beta) #mise a jour de alpha
        beta, t = beta_new, t_new

    return beta

# Exemple d'exécution
if __name__ == "__main__":
    np.random.seed(42)
    n, p = 100, 20
    X = np.random.randn(n, p)
    beta_vrai = np.random.randn(p)
    y = X @ beta_vrai + 0.1 * np.random.randn(n)  # Ajout d'un bruit gaussien

    lambda_ = 0.1
    L = np.linalg.norm(X.T @ X, 2) / n  # Approximation de la constante de Lipschitz
    num_iterations = 1000

    beta_estime = fista(X, y, lambda_, L, num_iterations)
    print("Coefficients estimés par FISTA :")
    print(beta_estime)
