import numpy as np
from scipy.stats import mode

from sklearn.base import RegressorMixin, BaseEstimator

class MeanRegressor(RegressorMixin):
    def __init__(self) -> None:
        super().__init__()
        
    # Predicts the mean of y_train
    def fit(self, X = None, y = None):
        '''
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
        Training data features
        y : array like, shape = (_samples,)
        Training data targets
        '''
        self._mean = np.mean(y)
        
        return self
        
        
    def predict(self, X = None):
        '''
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
        Data to predict
        '''
        return np.full(X.shape[0], fill_value=self._mean)