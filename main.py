import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
from fonction import GaussianClassifier, KNN, ParzenClassifier, Perceptron, Bagging, cross_validation, plot_visualisation, accuracy, plot_perceptron_decision
    
os.makedirs("images", exist_ok=True)

#%%
# Data 
datasets = {}
for tp in [1, 2, 3]:
    app = np.loadtxt(f"Data/data_tp{tp}_app.txt")
    pred = np.loadtxt(f"Data/data_tp{tp}_dec.txt")
    datasets[tp] = {
        'X_train': app[:, 1:3], 'y_train': app[:, 0],
        'X_test': pred[:, 1:3], 'y_test': pred[:, 0]
    }

#%%
for tp, data in datasets.items():
    print(f"\n{'='*50}\nTRAITEMENT DU TP{tp}\n{'='*50}")
    X_tr, y_tr = data['X_train'], data['y_train']
    X_te, y_te = data['X_test'], data['y_test']
    
    # --- 0. ANALYSES DESCRIPTIVES ---
    print(f"\n--- STATISTIQUES DESCRIPTIVES TP{tp} ---")
    plt.figure(figsize=(8, 6))
    
    for c in np.unique(y_tr):
        X_c = X_tr[y_tr == c]
        print(f"Classe {int(c)} -> Effectif: {len(X_c)}, Variances: [{np.var(X_c[:,0])}, {np.var(X_c[:,1])}]")
        
        # Trace les points et les ellipses 
        sns.scatterplot(x=X_c[:, 0], y=X_c[:, 1], label=f'Classe {int(c)}', alpha=0.7)
        sns.kdeplot(x=X_c[:, 0], y=X_c[:, 1], levels=2, alpha=0.5, linewidths=2)

    plt.title(f"TP{tp} - Visualisation des données et densités")
    plt.savefig(f"images/tp{tp}_descriptive.png")
    plt.close()


    # --- 1. ESTIMATION GAUSSIENNES ---
    print("\n--- ESTIMATION GAUSSIENNES ---")

    gauss_eucl = GaussianClassifier(metric="euclidienne")
    gauss_eucl.train(X_tr, y_tr)
    pred_eucl = gauss_eucl.predict(X_te)
    print(f'Accuracy Gaussien Euclidien : {accuracy(y_te, pred_eucl)}')
    ConfusionMatrixDisplay.from_predictions(y_te, pred_eucl, cmap="Blues")
    plt.title(f"TP{tp} - Gaussien Euclidien")
    plt.savefig(f"images/tp{tp}_gauss_eucl_cm.png")
    plt.close()
    plot_visualisation(gauss_eucl, X_tr, y_tr, title=f"images/tp{tp}_gauss_eucl_frontiere")
    
    gauss_mahal = GaussianClassifier(metric="mahalanobis")
    gauss_mahal.train(X_tr, y_tr)
    pred_mahal = gauss_mahal.predict(X_te)
    print(f'Accuracy Gaussien Mahalanobis : {accuracy(y_te, pred_mahal)}')
    ConfusionMatrixDisplay.from_predictions(y_te, pred_mahal, cmap="Blues")
    plt.title(f"TP{tp} - Gaussien Mahalanobis")
    plt.savefig(f"images/tp{tp}_gauss_mahal_cm.png")
    plt.close()
    plot_visualisation(gauss_mahal, X_tr, y_tr, title=f"images/tp{tp}_gauss_mahal_frontiere")
    
    gauss_logv = GaussianClassifier(metric="log-vraisemblance")
    gauss_logv.train(X_tr, y_tr)
    pred_logv = gauss_logv.predict(X_te)
    print(f'Accuracy Gaussien Log-Vraisemblance : {accuracy(y_te, pred_logv)}')
    ConfusionMatrixDisplay.from_predictions(y_te, pred_logv, cmap="Blues")
    plt.title(f"TP{tp} - Gaussien Log-Vraisemblance")
    plt.savefig(f"images/tp{tp}_gauss_logv_cm.png")
    plt.close()
    plot_visualisation(gauss_logv, X_tr, y_tr, title=f"images/tp{tp}_gauss_logv_frontiere")


    # --- 2. K PLUS PROCHES VOISINS ---
    print("\n--- ESTIMATION KNN ---")

    knn1 = KNN(k=1)
    knn1.train(X_tr, y_tr)
    pred_knn1 = knn1.predict(X_te)
    print(f'Accuracy 1-PPV : {accuracy(y_te, pred_knn1)}')
    ConfusionMatrixDisplay.from_predictions(y_te, pred_knn1, cmap="Blues")
    plt.title(f"TP{tp} - 1-PPV")
    plt.savefig(f"images/tp{tp}_1ppv_cm.png")
    plt.close()
    #plot_visualisation(knn1, X_tr, y_tr, title=f"images/tp{tp}_1ppv_frontiere")
    
    best_k, best_acc = 1, 0
    for k in [1, 3, 5, 7, 9, 11]:
        _, acc = cross_validation(KNN, X_tr, y_tr, n_folds=5, k=k)
        if acc > best_acc:
            best_acc, best_k = acc, k
            
    knn_best = KNN(k=best_k)
    knn_best.train(X_tr, y_tr)
    
    pred_knn_maj = knn_best.predict(X_te, choice="majority")
    print(f'Accuracy {best_k}-PPV Majorité : {accuracy(y_te, pred_knn_maj)}')
    ConfusionMatrixDisplay.from_predictions(y_te, pred_knn_maj, cmap="Blues")
    plt.title(f"TP{tp} - {best_k}-PPV Majorite")
    plt.savefig(f"images/tp{tp}_{best_k}ppv_maj_cm.png")
    plt.close()
    #plot_visualisation(knn_best, X_tr, y_tr, title=f"images/tp{tp}_{best_k}ppv_frontiere")
    
    pred_knn_imp = knn_best.predict(X_te, choice="impartial")
    print(f'Accuracy {best_k}-PPV Unanimité : {accuracy(y_te, pred_knn_imp)}')
    ConfusionMatrixDisplay.from_predictions(y_te, pred_knn_imp, cmap="Blues")
    plt.title(f"TP{tp} - {best_k}-PPV Unanimite")
    plt.savefig(f"images/tp{tp}_{best_k}ppv_imp_cm.png")
    plt.close()

    
    # --- 3. CLASSIFIEUR DE PARZEN ---
    print("\n--- ESTIMATION DE PARZEN ---")

    for kernel in ["uniforme", "gaussien"]:
        best_h, best_acc = 0.1, 0
        for h in [0.1, 0.5, 1.0, 1.5, 2.0, 3.0]:
            _, acc = cross_validation(ParzenClassifier, X_tr, y_tr, n_folds=5, h=h, kernel=kernel)
            if acc > best_acc:
                best_acc, best_h = acc, h
                
        parzen = ParzenClassifier(h=best_h, kernel=kernel)
        parzen.train(X_tr, y_tr)
        pred_parzen = parzen.predict(X_te)
        print(f'Accuracy Parzen {kernel} (h={best_h}) : {accuracy(y_te, pred_parzen)}')
        ConfusionMatrixDisplay.from_predictions(y_te, pred_parzen, cmap="Blues")
        plt.title(f"TP{tp} - Parzen {kernel} (h={best_h})")
        plt.savefig(f"images/tp{tp}_parzen_{kernel}_cm.png")
        plt.close()
        #plot_visualisation(parzen, X_tr, y_tr, title=f"images/tp{tp}_parzen_{kernel}_frontiere")
    
    
    # --- 4. PERCEPTRON ---
    print("\n--- PERCEPTRON ---")

    if tp == 1:
        classe_centrale = 5.0
    elif tp == 2:
        classe_centrale = 3.0  
    elif tp == 3:
        classe_centrale = 1.0


    # 5C - OvO (Concerne les 5 classes, s'arrête via max_iter si non séparable)
    perc_ovo_5c = Perceptron(strategy="one-vs-one", max_iter=1000)
    perc_ovo_5c.train(X_tr, y_tr)
    pred_ovo_5c = perc_ovo_5c.predict(X_te)
    print(f'Accuracy Perceptron OvO 5C : {accuracy(y_te, pred_ovo_5c)}')
    ConfusionMatrixDisplay.from_predictions(y_te, pred_ovo_5c, cmap="Blues")
    plt.title(f"TP{tp} - Perceptron OvO 5C")
    plt.savefig(f"images/tp{tp}_perc_ovo_5c_cm.png")
    plt.close()
    
    # Suppression de la classe centrale 
    m_tr, m_te = (y_tr != classe_centrale), (y_te != classe_centrale)
    X_tr_4c, y_tr_4c = X_tr[m_tr], y_tr[m_tr]
    X_te_4c, y_te_4c = X_te[m_te], y_te[m_te]
    
    # 4C - OvO (Pour comparer équitablement avec le OvA sur 4 classes)
    perc_ovo_4c = Perceptron(strategy="one-vs-one", max_iter=1000)
    perc_ovo_4c.train(X_tr_4c, y_tr_4c)
    pred_ovo_4c = perc_ovo_4c.predict(X_te_4c)
    print(f'Accuracy Perceptron OvO 4C : {accuracy(y_te_4c, pred_ovo_4c)}')
    ConfusionMatrixDisplay.from_predictions(y_te_4c, pred_ovo_4c, cmap="Blues")
    plt.title(f"TP{tp} - Perceptron OvO 4C")
    plt.savefig(f"images/tp{tp}_perc_ovo_4c_cm.png")
    plt.close()
    
    # 4C - OvA (Séparer une classe de toutes les autres)
    perc_ova_4c = Perceptron(strategy="one-vs-all", max_iter=1000)
    perc_ova_4c.train(X_tr_4c, y_tr_4c)
    plot_perceptron_decision(perc_ova_4c, X_tr_4c, y_tr_4c, title=f"images/tp{tp}_perc_ova_4c_frontiere")
    
    pred_ova_4c = perc_ova_4c.predict(X_te_4c)
    print(f'Accuracy Perceptron OvA 4C : {accuracy(y_te_4c, pred_ova_4c)}')
    ConfusionMatrixDisplay.from_predictions(y_te_4c, pred_ova_4c, cmap="Blues")
    plt.title(f"TP{tp} - Perceptron OvA 4C")
    plt.savefig(f"images/tp{tp}_perc_ova_4c_cm.png")
    plt.close()


    # --- 5. BAGGING ---
    print("\n--- BAGGING ---")
    best_n_perc, best_acc_perc = 5, 0
    for n in [5, 10, 20, 30]:
        _, acc = cross_validation(Bagging, X_tr, y_tr, n_folds=5, base_model_class=Perceptron, n_estimators=n, strategy="one-vs-one", max_iter=500)
        if acc > best_acc_perc:
            best_acc_perc, best_n_perc = acc, n
            
    bag_perc = Bagging(base_model_class=Perceptron, n_estimators=best_n_perc, strategy="one-vs-one", max_iter=500)
    bag_perc.train(X_tr, y_tr)
    pred_bag_perc = bag_perc.predict(X_te)
    print(f'Accuracy Bagging Perceptron (N={best_n_perc}) : {accuracy(y_te, pred_bag_perc)}')
    ConfusionMatrixDisplay.from_predictions(y_te, pred_bag_perc, cmap="Blues")
    plt.title(f"TP{tp} - Bagging Perceptron (N={best_n_perc})")
    plt.savefig(f"images/tp{tp}_bag_perc_cm.png")
    plt.close()

    best_n_knn, best_acc_knn = 5, 0
    for n in [5, 10, 20, 30]:
        _, acc = cross_validation(Bagging, X_tr, y_tr, n_folds=5, base_model_class=KNN, n_estimators=n, k=1)
        if acc > best_acc_knn:
            best_acc_knn, best_n_knn = acc, n
            
    bag_knn = Bagging(base_model_class=KNN, n_estimators=best_n_knn, k=1)
    bag_knn.train(X_tr, y_tr)
    pred_bag_knn = bag_knn.predict(X_te)
    print(f'Accuracy Bagging KNN (N={best_n_knn}) : {accuracy(y_te, pred_bag_knn)}')
    ConfusionMatrixDisplay.from_predictions(y_te, pred_bag_knn, cmap="Blues")
    plt.title(f"TP{tp} - Bagging KNN (N={best_n_knn})")
    plt.savefig(f"images/tp{tp}_bag_knn_cm.png")
    plt.close()