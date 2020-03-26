import numpy as np
import scipy.sparse as sp
from sklearn.utils import shuffle
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_raises_regex
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_equal
from sklearn.exceptions import NotFittedError
from sklearn import datasets
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from joblib import dump, load

def test_multi_output_classification_sample_weights():
    # weighted classifier
    Xw = [[1, 2, 3], [4, 5, 6]]
    yw = [[3, 2], [2, 3]]
    w = np.asarray([2., 1.])
    forest = RandomForestClassifier(n_estimators=10, random_state=1)
    clf_w = MultiOutputClassifier(forest)
    clf_w.fit(Xw, yw, w)
 
    # unweighted, but with repeated samples
    X = [[1, 2, 3], [1, 2, 3], [4, 5, 6]]
    y = [[3, 2], [3, 2], [2, 3]]
    forest = RandomForestClassifier(n_estimators=10, random_state=1)
    clf = MultiOutputClassifier(forest)
    clf.fit(X, y)
 
    X_test = [[0,0,1],[1.5, 2.5, 3.5], [3.5, 4.5, 5.5]]
    assert_almost_equal(clf.predict(X_test), clf_w.predict(X_test))
    yy = clf.predict(X_test)
    yv = clf_w.predict(X_test)
    print(type(yy), yy.shape)
    print(type(yv), yv.shape)
    print(yy)
    print(yv)

def test_multi_output_classification_sample_weights_save():
    # weighted classifier
    Xw = [[1, 2, 3], [4, 5, 6]]
    yw = [[3, 2], [2, 3]]
    w = np.asarray([2., 1.])
    forest = RandomForestClassifier(n_estimators=10, random_state=1)
    clf_w = MultiOutputClassifier(forest)
    clf_w.fit(Xw, yw, w)
    dump(clf_w,"myfirstpickle.pkl")
 
    # unweighted, but with repeated samples
    X = [[1, 2, 3], [1, 2, 3], [4, 5, 6]]
    y = [[3, 2], [3, 2], [2, 3]]
    forest = RandomForestClassifier(n_estimators=10, random_state=1)
    clf = MultiOutputClassifier(forest)
    clf.fit(X, y)
    dump(clf,"mysecondpickle.pkl")

def  test_multi_output_classification_sample_weights_load():
    clf_w = load("myfirstpickle.pkl")
    clf = load("mysecondpickle.pkl")

    X_test = [[0,0,1],[1.5, 2.5, 3.5], [3.5, 4.5, 5.5]]
    assert_almost_equal(clf.predict(X_test), clf_w.predict(X_test))
    yy = clf.predict(X_test)
    yv = clf_w.predict(X_test)
    print(type(yy), yy.shape)
    print(type(yv), yv.shape)
    print(yy)
    print(yv)
    
#test_multi_output_classification_sample_weights()
test_multi_output_classification_sample_weights_save()
test_multi_output_classification_sample_weights_load()


