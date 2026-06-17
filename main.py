# %% Ouverture des données
import numpy as np
import matplotlib.pyplot as plt 
from fonction import GaussianClassifier

#%% Ouverture des données
data = np.loadtxt("Data/data_tp3_app.txt")

# Séparation des X et des labels
labels = data[:, 0]        # Les classes (colonne 0)
X = data[:, 1:3]           # Les caractéristiques (colonnes 1 et 2)

# %% Estimation gaussienne 
# Distance Euclidien
model_eucl = GaussianClassifier(metric="euclidienne")
model_eucl.train(X, labels)

# Distance Mahalanobis
model_mahal = GaussianClassifier(metric="mahalanobis")
model_mahal.train(X, labels)

# %%% Prédiction sur de NOUVEAUX POINTS
# Points a prédire 
points_aleatoires = np.array([
    [4.0, 3.0], 
    [5.0, 3.0],
    [1.0, -2.0] 
])

# Obtenir les prédictions 
pred_eucl = model_eucl.predict(points_aleatoires)
pred_mahal = model_mahal.predict(points_aleatoires)
print(f"Prédictions Euclidienne : {pred_eucl}")
print(f"Prédictions Mahalanobis : {pred_mahal}")

# CORRECTION : Pour tester les distances brutes sur plusieurs points, on fait une petite compréhension de liste
dist_eucl_1 = [model_eucl.distance(pt, c=1.0) for pt in points_aleatoires]
dist_mahal_1 = [model_mahal.distance(pt, c=1.0) for pt in points_aleatoires]

print("\nDistances vers la classe 1 :")
for i, pt in enumerate(points_aleatoires):
    print(f"Point {pt} -> Eucl: {dist_eucl_1[i]:.4f} | Mahal: {dist_mahal_1[i]:.4f}")

# %%  Visualisation

def plot_visualisation(model, X_train, labels_train, X_new, frontiere=True):
    """
    Affiche les données d'entraînement, les barycentres et les nouveaux points
    """
    # thème  global
    plt.style.use('ggplot') 
    
    plt.figure(figsize=(10, 7))
    
    # Palette de couleurs harmonieuse (tab10)
    cmap = plt.get_cmap('tab10')
    
    # Frontière de décision
    if frontiere:
        x_min, x_max = plt.xlim() if plt.xlim() != (0.0, 1.0) else (X_train[:, 0].min() - 1, X_train[:, 0].max() + 1)
        y_min, y_max = plt.ylim() if plt.ylim() != (0.0, 1.0) else (X_train[:, 1].min() - 1, X_train[:, 1].max() + 1)
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                             np.linspace(y_min, y_max, 300))
        
        grille_points = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(grille_points)
        Z = Z.reshape(xx.shape)
        
        # Colorie les zones d'influence avec transparence 
        plt.contourf(xx, yy, Z, alpha=0.1, cmap='tab10')
        # Ligne de séparation
        plt.contour(xx, yy, Z, colors='black', linewidths=1.5, alpha=0.9)

    # Données d'entraînement et barycentres 
    for i, c in enumerate(model.classes):
        X_c = X_train[labels_train == c]
        couleur_classe = cmap(i) # On associe la couleur i à la classe c
        
        # Points d'entraînement
        plt.scatter(X_c[:, 0], X_c[:, 1], 
                    color=couleur_classe,
                    colors='black', linewidths=1.5,
                    alpha=0.7, linewidth=0.5, 
                    s=60, label=f'Classe {int(c)}')
        
        # Barycentres 
        barycentre = model.barycentres[c]
        plt.scatter(barycentre[0], barycentre[1], 
                    color=couleur_classe, s=400, linewidth=0.5, 
                    zorder=5, label=f'Barycentre {int(c)}')

    # Points inconnus
    if X_new.any(): 
        plt.scatter(X_new[:, 0], X_new[:, 1], 
                    c='black', marker='X', s=200, 
                    zorder=6, label='Nouveaux Points')
    
        # Petit badge esthétique pour le texte des nouveaux points
        for i, txt in enumerate(range(1, len(X_new) + 1)):
            plt.annotate(f" P{txt}", (X_new[i, 0] + 0.1, X_new[i, 1] + 0.1), 
                         fontsize=12, fontweight='bold', color='black',
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.7))
    
    # Finitions
    plt.title(f"Classification par distance - Métrique : {model.metric.capitalize()}", 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.xlabel('Caractéristique 1', fontsize=12, fontweight='bold')
    plt.ylabel('Caractéristique 2', fontsize=12, fontweight='bold')
    
    # Légende stylisée à l'extérieur
    legend = plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, shadow=True, fontsize=11)
    legend.get_frame().set_facecolor('white')
    
    plt.tight_layout()
    plt.show()
    
# Affichage pour le modèle avec distance Euclidienne
plot_visualisation(model_eucl, X_train=X, labels_train=labels, X_new=points_aleatoires)

# Et tu peux même le faire facilement pour le modèle Mahalanobis !
plot_visualisation(model_mahal, X_train=X, labels_train=labels, X_new=points_aleatoires)