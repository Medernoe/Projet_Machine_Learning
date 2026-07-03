import numpy as np

def euclidean_distance(X, Y): 
    return np.sqrt(np.sum((X - Y)**2))

def mahalanobis_distance(X, Y, cov, approx=True): 
    cov_inv = np.linalg.inv(cov)
    diff = X - Y
    det_cov = np.linalg.det(cov)

    # formule: diff^T * cov_inv * diff ou ln(det_cov)) + (diff^T * cov_inv * diff)
    stat = np.sqrt(diff.T @ cov_inv @ diff) if approx else np.log(det_cov) + (diff.T @ cov_inv @ diff)

    return stat

class GaussianClassifier:
    def __init__(self, metric="euclidienne"):
        """
        Initialise le classifieur.
        - metric: 'euclidienne', 'mahalanobis' ou 'log-vraisemblance'
        """
        # vérification de la distance choisie
        if metric not in ["euclidienne", "mahalanobis", "log-vraisemblance"]:
            raise ValueError("La métrique doit être 'euclidienne', 'mahalanobis' ou 'log-vraisemblance'.")
        
        # initialise les structures 
        self.metric = metric
        self.classes = None
        self.barycentres = {}
        self.cov = {} 

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
            
            if self.metric in ["mahalanobis", "log-vraisemblance"]:
                # Matrice de covariance / Divisé par n-1, correction de l'estimation
                self.cov[c] = np.cov(X_c, rowvar=False, bias=False)
                #self.cov_inverses[c] = np.linalg.inv(cov)
                  
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
            return mahalanobis_distance(x, mu, self.cov[c])
        
        elif self.metric == "log-vraisemblance":
            return mahalanobis_distance(x, mu, self.cov[c], False)


    def predict(self, P):
        """
        Prédit la classe pour un ou plusieurs points.
        - P: Tableau de pi points à prédire
        """
        
        # cas ou X est un point unique 
        if P.ndim == 1:
            P = P.reshape(1, -1)
            
        # liste des ti valeur prédite 
        T = []
        for p in P:
            # Calcule la distance de ce point à chaque classe
            distances = {c: self.distance(p, c) for c in self.classes}
            
            # Trouve la classe (clé du dictionnaire) avec la distance minimale
            predict_class = min(distances, key=distances.get)
            T.append(predict_class)
            
        return np.array(T)
    
class KNN: 
    def __init__(self, k=1):
        if k <= 0:
            raise ValueError("k doit etre un entier positif.")
        self.k = k
        self.X_train = None
        self.labels_train = None
        self.classes = None
        
    def train(self, X, labels): 
        """
        Stocke les données d'entraînement.
        """
        self.X_train = X
        self.labels_train = labels
        self.classes = np.unique(labels)
    
    def predict(self, P, choice="majority"): 
        """
        Prédit une classe pour un tableau de points P.
        """
        if choice not in ["majority", "impartiel"]:
            raise ValueError("choice doit être 'majority' ou 'impartiel'.")
        
        # cas ou P est un point unique 
        if P.ndim == 1:
            P = P.reshape(1, -1)
            
        predictions = []
        
        for p in P: 
            # Calcul des distances entre p et tous les points d'entraînement
            distances = [euclidean_distance(x, p) for x in self.X_train]
            
            # Récupération des k plus proches
            index_sorted = np.argsort(distances)
            labels_k_proches = self.labels_train[index_sorted[:self.k]]

            if choice == "majority": 
                unique_labels, occurrences = np.unique(labels_k_proches, return_counts=True)
                index_gagnant = np.argmax(occurrences)
                predictions.append(unique_labels[index_gagnant])
                
            elif choice == 'impartiel':
                unique_labels = np.unique(labels_k_proches)
                predictions.append(unique_labels[0] if len(unique_labels) == 1 else None)
                
        return np.array(predictions)
        

    
class ParzenClassifier:
    def __init__(self, h=1.0, kernel="gaussien"):
        """
        Initialise le classifieur à fenêtre de Parzen.
        - h : Largeur de la fenêtre (hyperparamètre, doit être > 0)
        - kernel : 'uniforme' ou 'gaussien'
        """
        if h <= 0:
            raise ValueError("L'hyperparamètre h doit être strictement positif.")
        if kernel not in ["uniforme", "gaussien"]:
            raise ValueError("Le noyau doit être 'uniforme' ou 'gaussien'.")
            
        self.h = h
        self.kernel = kernel
        self.X_train = None
        self.labels_train = None
        self.classes = None

    def train(self, X, labels):
        """
        Stocke les données d'entraînement.
        """
        self.X_train = X
        self.labels_train = labels
        self.classes = np.unique(labels)

    def compute_kernel(self, distances):
        """
        Calcule la valeur du noyau pour un tableau de distances.
        u = distance / h
        """
        u = distances / self.h
        
        if self.kernel == "gaussien":
            # Gaussien : exp(-0.5 * u^2)
            return np.exp(-0.5 * (u ** 2))
            
        elif self.kernel == "uniforme":
            # Uniforme : 1 si u <= 1 (à l'intérieur de la fenêtre), sinon 0
            return 1.0 if u >= 1 else 0 

    def predict(self, P):
        """
        Prédit une classe pour un ou plusieurs points P.
        """
        # Cas où P est un point unique
        if P.ndim == 1:
            P = P.reshape(1, -1)
            
        predictions = []
        
        for p in P:
            # Calcul des distances
            distances = np.array([euclidean_distance(x, p) for x in self.X_train])
            
            # Évalue le noyau 
            kernel_values = self.compute_kernel(distances)
            
            # Score
            scores = {}
            for c in self.classes:
                labels_train_c = (self.labels_train == c)
                scores[c] = np.sum(kernel_values[labels_train_c])
                
            # Trouve la classe avec le score maximum
            predict_class = max(scores, key=scores.get)
            predictions.append(predict_class)
            
        return np.array(predictions)
    
class Perceptron:
    def __init__(self, strategy="one-vs-one", max_iter=1000, lr=0.1):
        """
        Initialise le classifieur Perceptron.
        - strategy: 'one-vs-one' (K(K-1)/2 hyperplans) ou 'one-vs-all' (K hyperplans)
        - max_iter: Nombre maximum de boucles
        - lr: Taux d'apprentissage
        """
        if strategy not in ["one-vs-one", "one-vs-all"]:
            raise ValueError("La stratégie doit être 'one-vs-one' ou 'one-vs-all'.")
            
        self.strategy = strategy
        self.max_iter = max_iter
        self.lr = lr
        
        # Dictionnaire pour stocker les vecteurs de paramètres à optimiser
        self.weights = {} 
        self.classes = None

    def train_lr(self, X_transformed, y):
        """
        Apprentissage: Déterminer un hyperplan en utilisant la transformation normalisée.
        X_transformed contient déjà la composante (x, 1).
        """
        nb_individus, nb_features = X_transformed.shape
        
        # Initialisation du vecteur de paramètres (a_0, ... a_n) à optimiser
        W = np.zeros(nb_features) 
        
        best_W = np.copy(W)
        min_erreurs = nb_individus + 1

        # Transformation
        # Pour la classe +1 : on garde y1 = (x, 1)
        # Pour la classe -1 : on inverse pour obtenir y2 = (-x, -1)
        Y_transformed = np.zeros_like(X_transformed)
        for i in range(nb_individus):
            if y[i] == 1:
                Y_transformed[i] = X_transformed[i]   # (x, 1)
            else:
                Y_transformed[i] = -X_transformed[i] # (-x, -1)

        # Apprentissage
        for iteration in range(self.max_iter):
            erreurs = 0
            
            # On passe sur chaque individu transformé
            for i in range(nb_individus):
                y_i_norm = Y_transformed[i]
                
                # La somme des a_i * x_i + a_0 doit toujours être > 0, sinon erreur (grace a transformation)
                if np.dot(W, y_i_norm) <= 0:
                    W = W + self.lr * y_i_norm
                    erreurs += 1
            
            # Sauvegarde du meilleur vecteur de paramètres trouvé 
            if erreurs < min_erreurs:
                min_erreurs = erreurs
                best_W = np.copy(W)
                
            # modèle parfait sans erreurs 
            if erreurs == 0:
                break
                
        return best_W

    def train(self, X, labels):
        """
        Gère l'entraînement multi-classes en découpant le problème en sous-problèmes binaires.
        soit one-vs-one ou one-vs-all 
        """
        self.classes = np.unique(labels)
        
        # Transformation dans R^(d+1) pour avoir y=(x, 1), permet d'inclure a0 dans les poids
        # On ajoute la colonne de 1 à l'ensemble du dataset
        colonne_biais = np.ones((X.shape[0], 1))
        X_transformed = np.column_stack((X, colonne_biais))

        if self.strategy == "one-vs-one":
            # classes deux à deux : K(K-1)/2 hyperplans
            for i in range(len(self.classes)):
                for j in range(i + 1, len(self.classes)):
                    c1 = self.classes[i]
                    c2 = self.classes[j]
                    
                    pair_index = np.isin(labels, [c1, c2])
                    X_pair = X_transformed[pair_index]
                    labels_pair = labels[pair_index]
                                        
                    # Labels binaires standard, 2 classes (+1 / -1) pour la transformation dans train_lr
                    y_pair = np.where(labels_pair == c1, 1, -1)
                    
                    self.weights[(c1, c2)] = self.train_lr(X_pair, y_pair)

        elif self.strategy == "one-vs-all":
            # une classe contre toutes les autres : K hyperplans
            for c in self.classes:
                y_rest = np.where(labels == c, 1, -1)
                self.weights[c] = self.train_lr(X_transformed, y_rest)

    def predict(self, P):
        """
        Décision : positionnement du point inconnu par rapport aux surfaces.
        """
        if P.ndim == 1:
            P = P.reshape(1, -1)
            
        # Transformation des nouveaux points dans R^(d+1) : y=(x, 1)
        colonne_biais = np.ones((P.shape[0], 1))
        P_transformed = np.column_stack((P, colonne_biais))
        
        predictions = []

        for p in P_transformed:
            if self.strategy == "one-vs-one":
                votes = {c: 0 for c in self.classes}
                
                for (c1, c2), W in self.weights.items():
                    if np.dot(W, p) > 0:
                        votes[c1] += 1
                    else:
                        votes[c2] += 1
                        
                predict_class = max(votes, key=votes.get)
                predictions.append(predict_class)

            elif self.strategy == "one-vs-all":
                # Le point est assigné à la classe dont la fonction de décision est la plus grande
                scores = {c: np.dot(W, p) for c, W in self.weights.items()}
                predict_class = max(scores, key=scores.get)
                predictions.append(predict_class)

        return np.array(predictions)