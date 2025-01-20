import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.special import expit
from numpy import ndarray


class LogisticRegression():
    '''
    Модель логистической регрессии. Имеет следующие гиперпараметры:

    :param alpha: параметр регуляризации. 
                  Если равно 0, то регуляризация не происходит.
    :param lr: константа, на которую домножаем градиент при обучении
    :param max_iter: ограничение на кол-во итераций
    :param fit_intercept: указывает, следует ли добавить константу в признаки
    '''

    def __init__(self, 
                 alpha: float = 0.0, 
                 lr: float = 1e-4, 
                 max_iter: int = 1e5, 
                 fit_intercept: bool = True,
                 batch_size: int = 64) -> None:
        '''Создает модель и инициализирует параметры.'''

        self.alpha = alpha
        self.lr = lr
        self.max_iter = int(max_iter)
        self.fit_intercept = fit_intercept
        self.batch_size = batch_size

    @staticmethod
    def _sigmoid(x: ndarray) -> ndarray:
        return expit(x)

    def _add_intercept(self, X: ndarray) -> ndarray:
        X_copy = np.hstack([X, np.ones((X.shape[0], 1))])
        return X_copy

    def fit(self, X: ndarray, Y: ndarray):
        '''
        Обучает модель логистической регресии с помощью SGD,
        пока не выполнится self.max_iter итераций.

        :param X: матрица признаков
        :param Y: истинные метки
        '''

        if self.fit_intercept:  # добавляем свободный коэфициент
            X = self._add_intercept(X)
            
        Y_shifted = 2 * Y.copy() - 1  # используем модель с метками -1 и 1
            
        n, d = X.shape
            
        weights = np.zeros((d, ), dtype=X.dtype)

        for _ in range(self.max_iter):
            ids = np.random.choice(n, size=self.batch_size, replace=False)
            Xb = X[ids]
            Yb = Y_shifted[ids]
            
            grad = (Xb.T @ ((1 - self._sigmoid(Yb * (Xb @ weights))) * Yb)) / self.batch_size
            
            weights -= self.lr * grad
            
            #if self.fit_intercept:
            #    weights[:-1] -= self.lr * self.alpha * weights[:-1] 
            #else:
            weights -= self.lr * self.alpha * weights
            

        self.coef_ = weights[:-1] if self.fit_intercept else weights  # коэффициенты модели
        self.intercept_ = weights[-1:] if self.fit_intercept else None  # свободный коэффициент
        self.weights = weights

        return self

    def predict(self, X: ndarray) -> ndarray:
        '''
        Применяет обученную модель к данным 
        и возвращает точечное предсказание (оценку класса).

        :param X: матрица признаков
        :return: предсказание с размерностью (n_test, )
        '''

        if self.fit_intercept:
            X = self._add_intercept(X)
            
        proba = self._sigmoid(X @ self.weights)
        
        return (proba <= 0.5).astype(int)

    def predict_proba(self, X: ndarray) -> ndarray:
        '''
        Возвращает вероятность принадлежности классу 1.
        
        :param X: матрица признаков
        :return: массив вероятностей принадлежности классу 1
        '''
        if self.fit_intercept:
            X = self._add_intercept(X)

        prob_1 = self._sigmoid(X @ self.weights)
        return np.stack([1 - prob_1, prob_1], axis=-1)