import pandas as pd
from sklearn.base import ClassifierMixin

# There is an error in test system: y is included in X (that's a bad idea since y is target)

class RubricCityMedianClassifier(ClassifierMixin):
    def fit(self, X=None, y=None):
        self._rubric_city_median = pd.concat([X, y], axis=1).groupby(['modified_rubrics', 'city'])['average_bill'].median()
        return self
    
    def _apply_predict(self, x):
        return self._rubric_city_median[x['modified_rubrics'], x['city']]
    
    def predict(self, X:pd.DataFrame=None):
        return X.apply(func=self._apply_predict, axis=1).to_numpy()