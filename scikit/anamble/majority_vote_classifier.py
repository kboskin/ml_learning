import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder


class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):

    """
    classifiers = [LinearRegression, LogisticRegression, ...]
    vote = {'classlabel', 'probability', }
    weights = [weight for classifier]
    """
    def __init__(self, classifiers, vote='classlabel', weights=None):
        self.classifiers = classifiers
        self.named_classifiers = {key: value for key, value in _name_estimators(classifiers) }
        self.vote = vote
        self.weights = weights

    def fit(self, X, y):
        if self.vote not in ('probability', 'classlabel'):
            raise ValueError('wrong params')

        assert self.weights and self.weights != len(self.classifiers)

        self.labelenc_ = LabelEncoder()
        self.labelenc_.fit(y)
        self.classes_ = self.labelenc_.classes_
        self.classifiers_ = []

        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.labelenc_.transform(y))
            self.classes_.append(fitted_clf)
            return self

    def predict(self, X):
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:
            predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T

            maj_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=self.weights)), axis=1, arr=predictions)
            maj_vote = self.labelenc_.inverse_transform(maj_vote)
            return maj_vote

    def predict_proba(self, X):
        probas = np.asarray([cls.predict_proba(X) for cls in self.classifiers_])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba

    def get_params(self, deep=True):
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in self.named_classifiers.items():
                for key, value in step.get_params(deep=True).items():
                    out['%s__%s' % (name, key)] = value

            return out
