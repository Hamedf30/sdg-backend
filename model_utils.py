import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from typing import List, Tuple

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'sdg_model.joblib')
VEC_PATH = os.path.join(BASE_DIR, 'vectorizer.joblib')
MLB_PATH = os.path.join(BASE_DIR, 'mlb.joblib')


def train_and_save(texts: List[str], label_lists: List[List[str]]):
    """Train a One-vs-Rest multi-label classifier.
    label_lists: list of label lists (each inner list may be empty).
    Labels should be strings (e.g., '1', '2' or 'SDG1')."""
    vec = TfidfVectorizer(max_features=5000)
    X = vec.fit_transform(texts)

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(label_lists)

    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    clf.fit(X, Y)

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(vec, VEC_PATH)
    joblib.dump(mlb, MLB_PATH)
    return clf, vec, mlb


def load_model() -> Tuple[OneVsRestClassifier, TfidfVectorizer, MultiLabelBinarizer]:
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VEC_PATH) or not os.path.exists(MLB_PATH):
        raise FileNotFoundError('Model, vectorizer or mlb not found. Run training first.')
    clf = joblib.load(MODEL_PATH)
    vec = joblib.load(VEC_PATH)
    mlb = joblib.load(MLB_PATH)
    return clf, vec, mlb


def predict(text: str, threshold: float = 0.5) -> dict:
    clf, vec, mlb = load_model()
    X = vec.transform([text])

    # predict probabilities per class
    if hasattr(clf, 'predict_proba'):
        probs = clf.predict_proba(X)[0]  # shape (n_classes,)
    else:
        # some estimators do not implement predict_proba; fallback to decision_function
        probs = clf.decision_function(X)[0]
        probs = 1 / (1 + np.exp(-probs))

    classes = list(mlb.classes_)
    prob_dict = {str(c): float(p) for c, p in zip(classes, probs)}
    predicted = [str(c) for c, p in zip(classes, probs) if p >= threshold]

    # explain: top features per class using estimator coefficients
    feature_names = np.array(vec.get_feature_names_out())
    top_features = {}
    # OneVsRestClassifier stores estimators_ list
    for idx, cls in enumerate(classes):
        try:
            est = clf.estimators_[idx]
            coefs = est.coef_[0]
            contrib = coefs * X.toarray()[0]
            top_idx = np.argsort(contrib)[-10:][::-1]
            feats = list(feature_names[top_idx])
            top_features[str(cls)] = feats
        except Exception:
            top_features[str(cls)] = []

    return {
        'predicted_labels': predicted,
        'probabilities': prob_dict,
        'top_features': top_features,
    }
