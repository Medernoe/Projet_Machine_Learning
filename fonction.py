import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap


def euclidean_distance(X, Y): 
    return np.sqrt(np.sum((X - Y)**2))


def mahalanobis_distance(X, Y, cov, approx=True): 
    cov_inv = np.linalg.inv(cov)
    diff = X - Y
    det_cov = np.linalg.det(cov)

    # formule: diff^T * cov_inv * diff ou ln(det_cov)) + (diff^T * cov_inv * diff)
    stat = np.sqrt(diff.T @ cov_inv @ diff) if approx else np.log(det_cov) + (diff.T @ cov_inv @ diff)

    return stat


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def cross_validation(model_class, X, labels, n_folds=5, **model_params):
    """
    Effectue une validation croisée à n_folds sur un modèle donné.
    
    - base_model_class : La classe du modèle à évaluer (ex: Perceptron, KNN).
    - X : Les features du jeu de données complet.
    - labels : Les étiquettes du jeu de données complet.
    - n_folds : Le nombre de sous-échantillons, Par défaut 5.
    - model_params : Les hyperparamètres à passer au modèle.
    
    Retourne :
    - accuracies : Un tableau numpy contenant la précision de chaque fold.
    - mean_accuracy : La précision moyenne sur l'ensemble des fold.
    """
    
    # Mélange aléatoire des données
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    X_shuffled = X[indices]
    labels_shuffled = labels[indices]
    
    # n_folds
    X_folds = np.array_split(X_shuffled, n_folds)
    labels_folds = np.array_split(labels_shuffled, n_folds)
    
    accuracies = []
    
    # Apprentissage et de validation
    for i in range(n_folds):
        # Données test
        X_test = X_folds[i]
        labels_test = labels_folds[i]
        
        # Données d'entraînement 
        X_train = np.vstack([X_folds[j] for j in range(n_folds) if j != i])
        labels_train = np.concatenate([labels_folds[j] for j in range(n_folds) if j != i])
        
        # Instanciation du modèle avec ses paramètres spécifiques
        modele = model_class(**model_params)
        
        # Entraînement sur le set d'entraînement
        modele.train(X_train, labels_train)
        
        # Prédiction sur le set de test
        predictions = modele.predict(X_test)
        
        # Evaluation 
        fold_acc = accuracy(labels_test, predictions)
        accuracies.append(fold_acc)
        
    accuracies = np.array(accuracies)
    mean_accuracy = np.mean(accuracies)
    
    return accuracies, mean_accuracy

def plot_visualisation(model, X_train, labels_train, X_new=None, title="Visualisation"):
    plt.figure(figsize=(9, 6))

    # couleurs 
    classes = np.unique(labels_train)
    n_classes = len(classes)
    cmap_base = plt.get_cmap('Set1')
    colors = [cmap_base(i) for i in range(n_classes)]
    custom_cmap = ListedColormap(colors)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Grille de prédiction
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grille_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grille_points)
    Z = np.array(Z).reshape(xx.shape)
    Z_indices = np.vectorize(class_to_idx.get)(Z)
    bornes = np.arange(n_classes + 1) - 0.5
    plt.contourf(xx, yy, Z_indices, alpha=0.3, cmap=custom_cmap, levels=bornes)
    plt.contour(xx, yy, Z_indices, colors='black', linewidths=0.5, alpha=0.7, levels=bornes)

    # Points d'entraînement
    for i, c in enumerate(classes):
        data_class = (labels_train == c)
        plt.scatter(X_train[data_class, 0], X_train[data_class, 1], 
                    color=custom_cmap(i), edgecolors='black', label=f'Classe {int(c)}', alpha=0.8)

    # Barycentres (si le modèle en possède)
    if hasattr(model, 'barycentres'):
        for c, mu in model.barycentres.items():
            if c in class_to_idx: 
                color_idx = class_to_idx[c]
                plt.scatter(mu[0], mu[1], c=[custom_cmap(color_idx)], s=300, edgecolors='black', 
                            label=f'Barycentre {int(c)}' if c == classes[0] else "")

    plt.title(title, fontweight='bold', pad=15)
    plt.xlabel('Caractéristique 1')
    plt.ylabel('Caractéristique 2')
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig(title + ".png")
    plt.close()
    
    
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
        """
        Initialise le classifieur KNN.
        - k : Nombre de voisins (hyperparamètre, doit être > 0)
        """
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
        if choice not in ["majority", "impartial"]:
            raise ValueError("choice doit être 'majority' ou 'impartial'.")
        
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
                
            elif choice == 'impartial':
                unique_labels = np.unique(labels_k_proches)
                # -1 classe de rejet
                predictions.append(unique_labels[0] if len(unique_labels) == 1 else -1.0)
                
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
            # Uniforme : 1.0 si u <= 1 (à l'intérieur de la fenêtre), sinon 0.0
            return np.where(u <= 1, 1.0, 0.0)

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
        Apprentissage: Déterminer un hyperplan en utilisant la transformation.
        X_transformed déjà sous forme (x, 1).
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
    
    
class Bagging:
    def __init__(self, base_model_class, n_estimators=10, **model_params):
        """
        Initialise l'algorithme de Bagging.
        - base_model_class : La classe du modèle (ex: Perceptron, KNN).
        - n_estimators : Le nombre de classifieurs à entraîner dans l'ensemble.
        - model_params : Dictionnaire des hyperparamètres à passer au modèle de base.
        """
        if n_estimators <= 0:
            raise ValueError("Le nombre d'estimateurs doit être un entier positif.")
            
        self.base_model_class = base_model_class
        self.n_estimators = n_estimators
        self.model_params = model_params
        
        # Liste pour stocker les modèles entraînés
        self.estimators = [] 
        self.classes = None

    def train(self, X, labels):
        """
        Entraîne l'ensemble des modèles sur des échantillons Bootstrap.
        - X: Les features (ex: colonnes 1 et 2)
        - labels: Les labels (ex: colonne 0)
        """
        self.classes = np.unique(labels)
        n_samples = X.shape[0]
        
        # Réinitialise la liste si le modèle est réentraîné
        self.estimators = []
        
        # Entraînement des n_estimators modèles
        for _ in range(self.n_estimators):
            # Tirage aléatoire avec remise
            indices_bootstrap = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X[indices_bootstrap]
            labels_bootstrap = labels[indices_bootstrap]
            
            # Instanciation d'un nouveau modèle de base avec ses paramètres
            modele = self.base_model_class(**self.model_params)
            
            # Entraînement sur l'échantillon Bootstrap
            modele.train(X_bootstrap, labels_bootstrap)
            
            # Ajout à l'ensemble
            self.estimators.append(modele)

    def predict(self, P):
        """
        Prédit la classe pour un ou plusieurs points par vote majoritaire.
        - P: Tableau de points à prédire
        """
        # cas ou P est un point unique 
        if P.ndim == 1:
            P = P.reshape(1, -1)
            
        # Matrice pour stocker les prédictions de chaque modèle (n_estimators x n_points)
        toutes_predictions = np.zeros((self.n_estimators, P.shape[0]))
        
        for i, modele in enumerate(self.estimators):
            toutes_predictions[i] = modele.predict(P)
            
        predictions_finales = []
        
        # Vote majoritaire pour chaque point
        for j in range(P.shape[0]):
            # Récupère la colonne j correspondant aux votes des n_estimators pour le point j
            votes = toutes_predictions[:, j]
            
            # Compte les occurrences de chaque classe prédite
            unique_classes, occurrences = np.unique(votes, return_counts=True)
            
            # Détermine la classe gagnante
            index_gagnant = np.argmax(occurrences)
            predictions_finales.append(unique_classes[index_gagnant])
            
        return np.array(predictions_finales)

