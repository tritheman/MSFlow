import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def make_data(n_samples=2000, n_features=20, n_informative=10, class_sep=1.0, weights=None, random_state=42, test_size=0.2):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        weights=weights,
        class_sep=class_sep,
        random_state=random_state,
        shuffle=True,
    )
    cols = [f"f{i}" for i in range(n_features)]
    X = pd.DataFrame(X, columns=cols)
    y = pd.Series(y, name="target")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test