---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Predicting Breast Cancer With Multiple Classification Algorithms

```{code-cell} ipython3
:tags: [remove-cell]

import sys
sys.path.append( '../..' )
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.mean_cross_val_scores import mean_cross_val_scores
from src.plot_boxplot import boxplot_plotting
from src.plot_hist import plot_hist_overlay
%matplotlib inline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay
)
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    cross_validate,
    train_test_split
)
from myst_nb import glue
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
```

Authors: Tien Nguyen, Edison Le, Claudio Torres Cantú, Clichy Bazenga
## I. Summary

In this project, we perform data analysis to look for the most efficient model to predict whether it is benign and malignant tumour based on different measurements/traits observed from the data. We use various machine learning algorithms that classify whether a breast tumour is malignant or benign accordingly, and eventually decide on the best algorithm for that task based on recall score. The dataset is obtained from https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28original%29

## II. Introduction

### Background

Breast cancer is the development of cancerous tissue of the breast. A malignant (cancerous) tumour can destroy other healthy tissue surrounding it and the disease could potentially metastasize, causing numerous other health complications. However, a tumour might or might not be cancerous. A benign (non-cancerous) tumour will not cause as much harm to the patient compared to a malignant one. With that in mind, an early diagnosis of a **malignant tumour** would give both patients and health care providers valuable time to quickly devise various treatment plans to stop the progression of the disease. This is important especially for a silent killer like cancer, in which the earlier a correct diagnosis has been made, the drastically better the prognosis {cite}`MLbreastCancer, BreastCancer, DeepLearnCancer`.
This raises the question:

**Can we predict whether a tumor is malignant or benign based  on the different features observed from the tumour?**

### Dataset Description

The dataset we use: Breast Cancer Wisconsin (Original) Data Set, created by Dr. WIlliam H. Wolberg (physician), contains the following features with 699 observations {cite}`Database`:

1. Sample code number: id number
2. Clump Thickness: 1 - 10
3. Uniformity of Cell Size: 1 - 10
4. Uniformity of Cell Shape: 1 - 10
5. Marginal Adhesion: 1 - 10
6. Single Epithelial Cell Size: 1 - 10
7. Bare Nuclei: 1 - 10
8. Bland Chromatin: 1 - 10
9. Normal Nucleoli: 1 - 10
10. Mitoses: 1 - 10

With the relevant target class:

11. Class: (2 for benign, 4 for malignant)

+++ {"tags": []}

## III. Method & Results

### Exploratory Data Analysis

+++

Since the dataset did not come with column headers, we will first manually add them and load data. The headers will correspond to the order of their respective description above.

```{code-cell} ipython3
:tags: [remove-cell]

col_names = ["id", "clump", "unif_size", "unif_shape", "adhesion", "epi_size",
             "nuclei", "chromatin", "nucleoli", "mitoses", "class"]

dataset = pd.read_csv("../../data/raw/breast_cancer.txt", names=col_names, sep=",")
```

```{code-cell} ipython3
:tags: [remove-cell]

head = dataset.head()
glue("header", head)
```

```{glue:} header
```

```{code-cell} ipython3
:tags: [remove-cell]

dataset.info()
```

We see that the dataset uses "?" for missing data so we eliminate rows that contain "?". As shown above, all of the variables are numeric except for variables nuclei, we decide to transform it into type int for ease of data analysis later on. Finally, "id" feature does not appear to be useful for the prediction task; hence, it is dropped before carrying on to further analysis. 

```{code-cell} ipython3
:tags: [remove-cell]

dataset = dataset[(dataset != '?').all(axis=1)]
dataset['nuclei'] = dataset['nuclei'].astype(int)
dataset = dataset.drop(columns=["id"])
```

We also decide to replace benign class from 2 to 0 and malignant class from 4 to 1 since if we keep values of 2 and 4, it would be hard for predictive models to calculate accuracy, precision, and so on. 

```{code-cell} ipython3
:tags: [remove-cell]

dataset['class'] = dataset['class'].replace([2],0)
dataset['class'] = dataset['class'].replace([4],1) 
values = dataset['class'].value_counts(normalize = True)
glue("vals", values)
```

```{glue:} vals

```

There is an imbalance in the dataset between benign and malignant. We want to further investigate the malignant examples (class = 4); hence the classes should have same importance so recall would be the most appropriate and main metric for this project. This is due to the fact that in a medical diagnosis setting, false negative result for a disease like cancer will cause a lot more harm than a false positive result.

Then we split the data into training and testing sets (X_train, X_test, y_train, y_test) and extract names of numeric features for further exploration. 
 

```{code-cell} ipython3
:tags: [remove-cell]

train_df, test_df = train_test_split(dataset, test_size=0.3, random_state=123)
X_train = train_df.drop(columns=["class"])
X_test = test_df.drop(columns=["class"])

y_train = train_df["class"]
y_test = test_df["class"]
numeric_looking_columns = X_train.select_dtypes(include=np.number).columns.tolist()
train_df.info()
```

```{code-cell} ipython3
:tags: [remove-cell]

benign_cases = train_df[train_df["class"] == 0]
malignant_cases = train_df[train_df["class"] == 1]
print(len(numeric_looking_columns))
```

```{code-cell} ipython3
:tags: [remove-cell]

fig = plot_hist_overlay(df0=benign_cases, df1=malignant_cases,
                 columns=numeric_looking_columns, labels=["0 - benign", "1 - malignant"],
                 fig_no="1")
```

```{figure} ../../results/figures/hist_plot.png
---
height: 700px
name: Histograms
---
Histograms of Class Targets with respecto to each Explanatory Variable 
```



```{code-cell} ipython3
:tags: [remove-cell]

fig = boxplot_plotting(3,3,20,25,numeric_looking_columns,train_df,2)
```

```{figure} ../../results/figures/boxplot_plot.png
---
height: 700px
name: Boxplots
---
Boxplots of Class Targets with respecto to each Explanatory Variable 
```

Judging from the {ref}`histograms <Histograms>`, it is indicated that benign tumors tend to have smaller values for all the features listed in the dataset (clump, unif_size and so on) while malignant tumors have different tendencies for each features.

From the {ref}`boxplots <Boxplots>` above, chromatin and clump feature display the clearest differentiation between benign and malignant tumor. Benign tumors seem to have clump thickness and bland chromatin on the lower ends. These characteristics are also displayed clearly in the histograms. Moreover, it might not be as obvious in {ref}`boxplot <Boxplots>`, but in  {ref}`histogram <Histograms>`, the majority of benign tumors is on the lower end of the nuclei's spectrum while the majority of malignant tumors are on the upper end. This suggests tumors with higher values of clump thickness, chromatin and nuclei have a higher chance of being malignant. 

As for unif_size, unif_shape, adhesion and nucleoli, the {ref}`histogram <Histograms>` show that regarding to benign tumors, these features tend to range in small values while malignant tumors' ranges are more evenly spread out. This proposes tumors with smaller values of said features are more likely to be benign. But the proposal is not as strong as the one discussed above as the distinction is not as prominent.

Lastly, mitoses and epi_size feature indicate to have an overlap in values between benign and malignant tumors shown in  {ref}`histogram <Histograms>`. Hence, they might not be as useful in prediction task as other features. 

+++

```{code-cell} ipython3
:tags: [remove-cell]

numeric_transformer = StandardScaler()
ct = make_column_transformer(
    (numeric_transformer, numeric_looking_columns))
```

### Data analysis 
Since all features are numeric, we decide to scale our data to ensure that there is no bias presents when predicting results. 

Even though the main score we will be comparing when choosing the models is recall, we still want to look into accuracy and decision. Because between a model performs well on recall but have very low accuracy and precision and a model performs just a bit worse on recall but have excellent accuracy and precision, the latter model will still have an upper hand and be more preferable. 

```{code-cell} ipython3
:tags: [remove-cell]

scoring = [
    "accuracy",
    "f1",
    "recall",
    "precision",
]
```

We decide to test 3 models: Decision Tree, kNN and Logistic Regression. Decision Tree, kNN, Logistic Regression are simple models with fast fit_time and moderate precision and accuracy and suitable for the classification/prediction task.

```{code-cell} ipython3
:tags: [remove-cell]

np.random.seed(123)

pipe_knn = make_pipeline(ct, KNeighborsClassifier(n_neighbors=5))
pipe_dt = make_pipeline(ct, DecisionTreeClassifier(random_state=123))
pipe_reg = make_pipeline(ct, LogisticRegression(max_iter=100000))

classifiers = {
    "kNN": pipe_knn,
    "Decision Tree": pipe_dt,
    "Logistic Regression" : pipe_reg
}

results = {}

for (name, model) in classifiers.items():
    results[name] = mean_cross_val_scores(
        model, 
        X_train, 
        y_train, 
        return_train_score=True, 
        scoring = scoring
    )

pd.DataFrame(results).T

table = pd.read_csv("../../results/tables/cross_val.csv", index_col=0, sep=",")
glue("df_table",table)
glue("knn-based-rec", np.around(table.iloc[0]["test_recall"], 3))
glue("knn-based-prec", np.around(table.iloc[0]["test_precision"], 3))
glue("knn-based-acc", np.around(table.iloc[0]["test_accuracy"], 3))
glue("logit-prec", np.around(table.iloc[2]["test_precision"], 3))
glue("logit-rec", np.around(table.iloc[2]["test_recall"], 3))

```

```{glue:} df_table
```
As shown above, kNN model is the best performing model with the highest `test_recall score`, {glue:text}`knn-based-rec`, and high `test_precision` and `test_accuracy`, {glue:text}`knn-based-prec` and {glue:text}`knn-based-acc` respectively. Even though Logistics Regression has higher precision score {glue:text}`logit-prec` than KNN's, its recall score is lower than those of kNN ({glue:text}`logit-rec` < {glue:text}`knn-based-rec` respectively). Overall, kNN model still performs better than Logistics Regression. Decision Tree performs the worst since it displays the lowest scores among 3 models. 

After choosing the most efficient model, kNN, we move onto tuning its hyperparameters to increase its performance. We decide to tune n_neighbors which determines the number of neighbors k and weights which determines weight function used in prediction. 

```{code-cell} ipython3
:tags: [remove-cell]

np.random.seed(123)

search = GridSearchCV(pipe_knn,
                      param_grid={'kneighborsclassifier__n_neighbors': range(1, 50),
                                  'kneighborsclassifier__weights': ['uniform', 'distance']},
                      cv=10, 
                      n_jobs=-1,  
                      scoring="recall", 
                      return_train_score=True)

search.fit(X_train, y_train)
print(search.best_score_)
print(search.best_params_)
```

```{code-cell} ipython3
:tags: [remove-cell]
tuned = pd.read_csv("../../results/tables/tuned_para.csv", index_col=0, sep=",")
glue("tuned_para", tuned)
glue("para1", tuned.iloc[0]["kneighborsclassifier__n_neighbors"])
glue("para2", tuned.iloc[0]["kneighborsclassifier__weights"])
glue("recall", np.around(tuned.iloc[0]["knn_best_score"], 3))
```

```{glue:} tuned_para
```

After tuning hyperparameters, we successfully increase recall score from {glue:text}`knn-based-rec` to {glue:text}`recall` with hyperparameters `n_neighbors=` {glue:text}`para1` and `weights=` {glue:text}`para2`.  

+++

### Model on test set 

+++

We then apply tuned hyperparameters, `n_neighbors=` {glue:text}`para1` and `weights=` {glue:text}`para2`, to kNN model and test the model's generalization on test set. 

We also print a result plot and plot a confusion matrix for X_test and y_test to visualize the test results. 

```{code-cell} ipython3
:tags: [remove-cell]

pipe_knn_tuned = make_pipeline(ct,KNeighborsClassifier(n_neighbors=5, weights='uniform'))
pipe_knn_tuned.fit(X_train, y_train)
pipe_knn_tuned.score(X_test, y_test)
```

```{code-cell} ipython3
:tags: [remove-cell]

print(classification_report(
    y_test, pipe_knn_tuned.predict(X_test), target_names=["benign", "malignant"]))
```


```{code-cell} ipython3
:tags: [remove-cell]

predictions = pipe_knn_tuned.predict(X_test)
cm = confusion_matrix(y_test, predictions, labels=pipe_knn_tuned.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=pipe_knn_tuned.classes_)
disp.plot()
plt.title("Figure 3: Confusion Matrix")
```


```{code-cell} ipython3
:tags: [remove-cell]
preds = pd.read_csv("../../results/tables/classification_report.csv", index_col=0, sep=",")
glue("predict", preds)
glue("fin-avg-acc", np.around(preds.iloc[2]["recall"], 3))
glue("fin-avg-prec", np.around(preds.iloc[4]["precision"], 3))
glue("fin-avg-rec", np.around(preds.iloc[4]["recall"], 3))
glue("fin-avg-f1", np.around(preds.iloc[4]["f1-score"], 3))
glue("fin-mal-rec", np.around(preds.iloc[1]["recall"], 3))
glue("fin-be-rec", np.around(preds.iloc[0]["recall"], 3))
```

```{glue:} predict
```

```{figure} ../../results/figures/confusion_matrix.png
---
height: 500px
name: Conf-Matrix
---
Confusion matrix for Algorithms Classification
```

After applying the model to the test set, we obtain the score of {glue:text}`fin-avg-acc` which indicates that this model generalizes well for prediction task. Moreover, from the plot above {ref}`(Figure 3) <Conf-Matrix>`, we can see that out of **205** observations in test set, we only falsely predict **1** tumor to be benign and **2** tumors to malignant. This aligns with our decision at the beginning of the project which is to optimize recall score of the model. 

With the recall score of {glue:text}`fin-mal-rec` for malignant tumors and {glue:text}`fin-be-rec` for benign tumors reported above, we have statistical evidence that the model generalizes well for the prediction task.  

+++

## IV. Summary of results and discussion  

```{code-cell} ipython3
:tags: [remove-cell]

print(
    classification_report(
        y_test, pipe_knn_tuned.predict(X_test), target_names=["benign", "malignant"]
    )
)

glue("fin-avg-acc-per", np.around(preds.iloc[2]["recall"], 3)*100)
glue("fin-avg-prec-per", np.around(preds.iloc[4]["precision"], 3)*100)
glue("fin-avg-rec-per", np.around(preds.iloc[4]["recall"], 3)*100)
glue("fin-avg-f1-per", np.around(preds.iloc[4]["f1-score"], 3)*100)
```


```{glue:} predict
```

### Summary of Findings

The boxplots and histograms allow us to see how the displayed data is distributed across our dataset and how they differentiate benign and malignant tumors. We believe certain variables will give us significantly more information on how to correctly classify a tumor. Such features which can play a decisive factor in prediction task are: 

Chromatin (Figures {ref}` 1.7 <Histograms>` & {ref}` 2.7 <Boxplots>`): It is a variable which at first sight could serve us as a way to determine the class label for a given test example. The problem with this variable is the intersection between 2 & 4 in the x axis for figure {ref}` 1.7 <Histograms>`. This is problematic because the probability of falsely predicting an observation increases if the value of chromatin of said observation lies within that given interval.

Nucleoli (Figures{ref}` 1.8 <Histograms>`  & {ref}` 2.8 <Boxplots>`): This feature plays an important role in classifying tumor types as shown in both plots. Observing {ref}` 1.8 <Histograms>` we notice that benign tumors tend to have a nucleoli value in between 0 and 2 (for discrete values), and we primarily see malignant tumors for values greater than 2. This is further supported in {ref}` 2.8 <Boxplots>`, where we see that only outliers from benign tumors are entangled with the malignant tumor's box and lower whisker.

Other variables like nuclei or adhesion can also be useful, but if we were able to set them only for certain values of the "x" variable. The problem is that if we, for example, take nuclei as the decisive rule to classify, we would have almost a 50/50 chance of correctly classifying it if "x" was between 2 and 4 approximately. 

By comparing cross-val scores (recall scores) among kNN, Decision Tree, and Logistics Regression, we found that kNN performs the best with highest recall score. By using `GridSearchCV` to computationally optimize the hyperparameters for `KNeighborsClassifier`, it turns out our best model is `KNeighborsClassifier` with `n_neighbors=` {glue:text}`para1` and `weights=` {glue:text}`para2`. This model achieved an astounding {glue:text}`fin-avg-acc-per`**% overall accuracy**, {glue:text}`fin-avg-rec-per`**% recall**, {glue:text}`fin-avg-prec-per`**% precision and an F1-score of** {glue:text}`fin-avg-f1-per`**%** when deploying on test set. This means that the model generalized very well for this prediction problem, and with such a high recall we also achieved our goal of maximizing the True Positive and minimizing the False Negative instances, which would be of tremendous importance when it comes to computational-aided medical diagnostic.

### Impacts of our Findings
We hope to aid other researchers and medical professionals in breast cancer diagnostic process and treatment with our results. We also hope to inspire other researchers to build upon what we have to continually improve this machine learning model, or take on a different perspective and build other technologies that will hopefully be more powerful and geared towards the task than what we developed.

### Future Questions and Research Area
1. Are there any other features related to cancer tissue, either unique to breast cancer or universal to all types of cancer, that can further improve the accuracy metrics of machine learning models?
1. Is there a better machine learning algorithm that could be seen as the most "optimal" to this prediction task?
1. Develop a better systematic approach/procedure between medical professional and these technologies to diagnose breast cancer more efficiently.
1. Research into developing better technologies, not limited to AI/ML, to aid with breast cancer (or any cancer) diagnostic.

## V. References
```{bibliography} ../../doc/jbook/references.bib
:style: plain
```
