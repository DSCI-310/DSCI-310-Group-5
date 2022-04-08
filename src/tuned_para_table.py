# import pandas as pd
# import sklearn
# import numpy as np


# def tuned_para_table(search, X_train, y_train):
#     """
#     A function which returns a panda dataframe of tuned hyperparameters
#     and its best score given GridSearchCV object fitted X_train and y_train
#     -------------------
#     PARAMETERS:
#     search: A sklearn.model_selection._search.GridSearchCV that has been
#     specified estimator, param_grid, **kwargs
#     X_train : numpy array or pandas DataFrame/Series
#         X in the training data
#     y_train : numpy array or pandas DataFrame/Series
#         y in the training data
#     --------------------
#     REQUISITES:
#     X_train, y_train must at least n_splits (specified in cv in search)
#     observations for each target class.
#     search must be GridSearchCV object that is clearly specified with
#     estimator, param_grid, cv, and so on.
#     --------------------
#     RETURNS:
#     Returns a pandas.core.frame.DataFrame object that specifies
#     the tuned hyperaparameters and the best score produced by GridSearchCV
#     --------------------
#     Examples

#     search = GridSearchCV(KNeighborsClassifier(),
#                       param_grid={'kneighborsclassifier__n_neighbors':
#                       range(1, 10)},
#                       cv=10, 
#                       n_jobs=-1,  
#                       scoring="recall", 
#                       return_train_score=True)
#     --------
#     tuned_para_table(search, X_train, y_train)
#     """
#     if not isinstance(search, sklearn.model_selection._search.GridSearchCV):
#         raise TypeError("'search' should be of type GridSearchCV")
#     if not isinstance(X_train, (pd.core.series.Series,
#                                 pd.core.frame.DataFrame, np.ndarray)):
#         raise TypeError("'X_train' should be of type np.array or pd.Dataframe")
#     if not isinstance(y_train, (pd.core.series.Series,
#                                 pd.core.frame.DataFrame, np.ndarray)):
#         raise TypeError("'y_train' should be of type np.array or pd.Dataframe")
#     search.fit(X_train, y_train)
#     best_score = search.best_score_.astype(type('float', (float,), {}))
#     tuned_para = pd.DataFrame.from_dict(search.best_params_, orient='index')
#     tuned_para = tuned_para.rename(columns = {0 : "Value"})
#     tuned_para = tuned_para.T
#     tuned_para['best_score'] = best_score
#     return tuned_para