from __future__ import division, print_function, absolute_import

from .interface import Classifier, Regressor
from .sklearn import SklearnClassifier, SklearnRegressor
from .matrixnet import MatrixNetClassifier, MatrixNetRegressor

try:
    from .tmva import TMVAClassifier, TMVARegressor
except:
    pass


from .xgboost import XGBoostClassifier, XGBoostRegressor
