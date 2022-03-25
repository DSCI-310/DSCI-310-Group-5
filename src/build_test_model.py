import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from mean_cross_val_scores import mean_cross_val_scores
from sklearn.compose import make_column_transformer


scoring = [
    "accuracy",
    "f1",
    "recall",
    "precision",
]

results = {}


def build_test_model(train_df, test_df, cross_val_output, tuned_para_output,
                     classification_output, confusion_matrix_output):
    np.random.seed(123)
    train_df = pd.read_csv(train_df) 
    test_df = pd.read_csv(test_df)
    X_train = train_df.drop(columns=["class"])
    X_test = test_df.drop(columns=["class"])
    y_train = train_df["class"]
    y_test = test_df["class"]
    numeric_looking_columns = X_train.select_dtypes(
        include=np.number).columns.tolist()
    numeric_transformer = StandardScaler()
    ct = make_column_transformer((numeric_transformer, numeric_looking_columns))
    pipe_knn = make_pipeline(ct, KNeighborsClassifier(n_neighbors=5))
    pipe_dt = make_pipeline(ct, DecisionTreeClassifier(random_state=123))
    pipe_reg = make_pipeline(ct, LogisticRegression(max_iter=100000))
    classifiers = {
        "kNN": pipe_knn,
        "Decision Tree": pipe_dt,
        "Logistic Regression" : pipe_reg}

    #cross_val_scores_for_models
    for (name, model) in classifiers.items():
        results[name] = mean_cross_val_scores(
        model,
        X_train,
        y_train,
        return_train_score=True,
        scoring = scoring)
    cross_val_table = pd.DataFrame(results).T
    cross_val_table.to_csv(cross_val_output) 
    
    #tune hyperparameters 
    np.random.seed(123)
    search = GridSearchCV(pipe_knn,
                          param_grid={'kneighborsclassifier__n_neighbors': range(1,50),
                                      'kneighborsclassifier__weights': ['uniform', 'distance']},
                          cv=10, 
                          n_jobs=-1,  
                          scoring="recall", 
                          return_train_score=True)

    search.fit(X_train, y_train)
    best_score = search.best_score_.astype(type('float', (float,), {}))
    tuned_para = pd.DataFrame.from_dict(search.best_params_, orient='index')
    tuned_para = tuned_para.rename(columns = {0:"Value"})
    tuned_para = tuned_para.T
    tuned_para['knn_best_score'] = best_score
    tuned_para.to_csv(tuned_para_output)

    #model on test set 
    pipe_knn_tuned = make_pipeline(ct,KNeighborsClassifier(
        n_neighbors=search.best_params_['kneighborsclassifier__n_neighbors'], 
        weights=search.best_params_['kneighborsclassifier__weights']))
    pipe_knn_tuned.fit(X_train, y_train)

    #classification report 
    report = classification_report(y_test, pipe_knn_tuned.predict(X_test), 
                                   output_dict=True, target_names=["benign", "malignant"])
    report = pd.DataFrame(report).transpose()
    report.to_csv(classification_output)

    #confusion matrix 
    cm = confusion_matrix(y_test, pipe_knn_tuned.predict(X_test), labels=pipe_knn_tuned.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe_knn_tuned.classes_)
    disp.plot()
    plt.savefig(confusion_matrix_output)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build and test model")
    parser.add_argument("train_df", type=str, help="Path to train_df")
    parser.add_argument("test_df", type=str, help="Path to test_df")
    parser.add_argument("cross_val_output", type=str, help="Path to cross val scores output")
    parser.add_argument("tuned_para_output", type=str, help="Path to tuned parameters output")
    parser.add_argument("classification_output", type=str, help="Path to classification report output")
    parser.add_argument("confusion_matrix_output", type=str, help="Path to confusion matrix output")
    args = parser.parse_args()
    build_test_model(args.train_df, args.test_df, args.cross_val_output, args.tuned_para_output,
                    args.classification_output, args.confusion_matrix_output)
