# Lasso_Regression_Project

# 📌 Lasso Regression - Optimisation Numérique 📊

## 📖 Introduction
Ce projet explore **l'optimisation numérique appliquée à la régression Lasso** en utilisant trois algorithmes d'optimisation :
- **ISTA (Iterative Shrinkage Thresholding Algorithm)**
- **FISTA (Fast Iterative Shrinkage Thresholding Algorithm)**
- **CGDA (Coordinate Gradient Descent Algorithm)**

L'objectif est d'analyser et de comparer leurs performances en termes de **vitesse de convergence et précision des prédictions**.

Ce travail est principalement basé sur le papier scientifique suivant :
> **Zhao, Y., & Huo, X. (2023).** *A Survey of Numerical Algorithms that can Solve the Lasso Problems*.  
> 📄 [Lien vers l'article](https://arxiv.org/abs/2303.03576)

---

## 📊 Jeu de données utilisé
Nous utilisons le **dataset "Communities and Crime"** provenant de la UCI Machine Learning Repository.  
Ce jeu de données regroupe des informations socio-économiques, démographiques et des statistiques criminelles.

📌 **Lien du dataset :**  
🔗 [Communities and Crime Dataset](https://archive.ics.uci.edu/ml/datasets/communities+and+crime)

### **📌 Description rapide :**
- **Nombre d'instances :** 1994  
- **Nombre d'attributs :** 128  
- **Tâche :** Prédiction du **taux de crimes violents par habitant**  
- **Types de variables :** socio-économiques, démographiques, statistiques policières  

Nous avons appliqué **un pré-traitement** pour gérer les valeurs manquantes et standardiser les données.

---

## 🛠️ Implémentation et Analyse
Les algorithmes ont été implémentés en **Python**, et leur performance a été mesurée en utilisant :
- **Validation croisée (5-fold)**
- **Erreur Quadratique Moyenne (MSE)**
- **Comparaison des vitesses de convergence**

Chaque algorithme a été testé avec différentes valeurs du paramètre de régularisation **λ** (0.1, 1, 10).

---

## 📈 Résultats et Comparaison
Les résultats obtenus montrent que **FISTA offre une convergence plus rapide** par rapport à ISTA et CGDA.  
Cependant, CGDA obtient **le meilleur RMSE pour λ=0.1**, ce qui indique une meilleure généralisation sur ce dataset.

👉 **Les détails de la comparaison et les visualisations sont disponibles dans le rapport final.**

---

## 📌 Comment exécuter le projet
1. **Cloner le repository** :
   ```bash
   git clone https://github.com/kchaouamira0/Lasso_Regression_Project.git
   cd Lasso_Regression_Project
