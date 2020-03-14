
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval
from sklearn.model_selection import StratifiedKFold, cross_val_score


# ----------------------------------------------------------------------------------
#               				GridSearch
# ----------------------------------------------------------------------------------

def gridsearch(X_train,y_train,classifer, param_grid, n_iter, cv, filename='./results'):
    grid  = RandomizedSearchCV(classifer, param_grid, n_iter = n_iter, cv = cv, scoring = "accuracy", n_jobs = -1,random_state=1)
    grid.fit(X_train,y_train)
    opt_parameters = grid.best_params_
    print(grid.best_params_)
    
    params_file = open(filename, 'w')
    params_file.write(str(grid.best_params_))
    params_file.close()
    return opt_parameters

