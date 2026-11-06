"""Classical baseline models (scikit-learn).

All models except 'SVM (RBF, 30 features)' receive exactly the same reduced,
[0, pi]-scaled PCA features as the quantum models, so the comparison is fair.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from .preprocess import SEED


def tuned_rbf_svm(seed=SEED):
    """RBF-kernel SVM with C and gamma chosen by 5-fold stratified CV."""
    grid = {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1, 1]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return GridSearchCV(SVC(kernel="rbf"), grid, cv=cv, scoring="accuracy")


def build_classical_models(seed=SEED):
    """Return {name: unfitted estimator} for the reduced-feature baselines."""
    return {
        "SVM (RBF)": tuned_rbf_svm(seed),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=seed),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "MLP": MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000, random_state=seed),
    }
