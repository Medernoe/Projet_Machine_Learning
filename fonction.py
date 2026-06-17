import numpy as np

def euclidean_distance(X, Y): 
    return np.sqrt(np.sum((X - Y)**2))

def mahalanobis_distance(X, Y, cov_inv): 
    diff = X - Y
    # formule: diff^T * cov_inv * diff
    return np.sqrt(diff.T @ cov_inv @ diff)

class GaussianClassifier:
    def __init__(self, metric="euclidienne"):
        """
        Initialise le classifieur.
        - metric: 'euclidienne' ou 'mahalanobis'
        """
        # vérification de la distance choisie
        if metric not in ["euclidienne", "mahalanobis"]:
            raise ValueError("La métrique doit être 'euclidienne' ou 'mahalanobis'.")
        
        # initialise les structures 
        self.metric = metric
        self.classes = None
        self.barycentres = {}
        self.cov_inverses = {} 

    def train(self, X, labels):
        """
        Entraîne le modèle : trouve les classes, calcule les barycentres et matrices de covariance.
        - X: Les features (ex: colonnes 1 et 2)
        - labels: Les labels (ex: colonne 0)
        """
        
        #trouves les labels 
        self.classes = np.unique(labels)
        
        # entrainement 
        for c in self.classes:
            # Extraction des données de la classe c
            X_c = X[labels == c]
            
            # Calcul du barycentre (moyenne sur chaque colonne)
            self.barycentres[c] = np.mean(X_c, axis=0)
            
            if self.metric == "mahalanobis":
                # Matrice de covariance / Divisé par n-1, correction de l'estimation
                cov = np.cov(X_c, rowvar=False, bias=True)
                self.cov_inverses[c] = np.linalg.inv(cov)
                  
    def distance(self, x, c):
        """
        Calcule la distance entre un point x et le barycentre de la classe c.
        - x: point à n dimension 
        - c: barycentre d'une classe données 
        """
        mu = self.barycentres[c]
        
        if self.metric == "euclidienne":
            return euclidean_distance(x, mu)
            
        elif self.metric == "mahalanobis":
            return mahalanobis_distance(x, mu, self.cov_inverses[c])

    def predict(self, X):
        """
        Prédit la classe pour un ou plusieurs points.
        - X: Tableau de xi points à prédire
        """
        
        # cas ou X est un point unique 
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        # liste des ti valeur prédite 
        T = []
        for x in X:
            # Calcule la distance de ce point à chaque classe
            distances = {c: self.distance(x, c) for c in self.classes}
            
            # Trouve la classe (clé du dictionnaire) avec la distance minimale
            predict_class = min(distances, key=distances.get)
            T.append(predict_class)
            
        return np.array(T)
    

class KNN: 
    def __init__(self, k = 1):
        """
        Initialise le knn.
        - k : taille du voisinage 
        """
        # vérification du k 
        if k < 0:
            raise ValueError("k doit etre un entier positif.")
        
        # initialise les structures 
        self.k = k
        self.classes = None
        
    def train(self, X, labels): 
        pass
    
    def predict(self, X): 
        pass