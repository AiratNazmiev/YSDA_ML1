import numpy as np
from sklearn.base import RegressorMixin

class CityMeanRegressor(RegressorMixin):
    def fit(self, X=None, y=None):
        self._cities = list(X['city'].unique())
        self._city_mean = {}
        for c in self._cities:
            self._city_mean[c] = y[X['city'] == c].mean()
          
    def predict(self, X=None):
        y = np.full(X.shape[0], fill_value=np.inf)
        for c in self._cities:
            y[X['city'] == c] = self._city_mean[c]
        
        return y