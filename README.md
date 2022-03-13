# DSCI-310-Group-5

## Team member:
- Tien Nguyen
- Claudio Torres Cantú
- Edison Le
- Clichy Bazenga Mwangombe


## Summary
The project seeks to provide a solution to the prediction problem of spotting benign and malignant tumors, which comes from the question "Is there a way to efficiently classify whether a tumor is malignant or benign with high accuracy,  given a set of different features observed from the tumor in its development stage?". 
Such problem was resolved using a predictive model. Our initial hypothesis was that it is possible to do so yet it would have a high error rate due to tumors features' variations. 
After performing EDA, such as summary statistics and data cleaning and visualization, we were able to spot some clear distinctions between benign and malignant tumors in some features.
We then tested multiple different classification models and arrived at a K-Nearest-Neighbor model with tuned hyperparameters with very good accuracy, recall, precision and f1 score. 

## Instructions for Execution
The project was developed in Python, specifically in Python version 3.9.10
This project relies heavily in Python Packages related to Machine Learning and Scientific Computation in Python. Pandas, NumPy, Matplotlib, Seaborn and Scikit-Learn. 
The dependencies needed are:

|Dependency  |   Version|
|------------|----------|
|Pandas      |   1.3.4  |
|Numpy       |   1.20.3 |
|matplotlib  |   3.4.3  |
|seaborn     |   0.11.2 |
|scikit-learn|   0.24.2 |
|pytest      |   6.2.4  |

In order to run the code, we advise you to use the Dockerfile listed in the main branch to have a suitable environment and avoid any problems with dependencies. The code is intended to run in JupyterLab.

Using the command to pull the repository to your local machine:

`docker pull nhantien/dsci310group5:v0.3.0`

This command pulls the *nhantien/dsci310group5* image tagged *v0.3.0* from Docker Hub if it is not already present on the local host. It then starts a container running a Jupyter Server and exposes the container’s internal port 8888 to port 8888 of the host machine:

`docker run -it --rm -p 8888:8888 nhantien/dsci310group5:v0.3.0`

Visiting http://<hostname>:10000/?token=<token> in a browser loads JupyterLab, where:

* `<hostname>` is the name of the computer running Docker

* `<token>` is the secret token printed in the console.

The container will be cleaned up and removed after the Jupyter Server exits with the flag `--rm`. Simply remove the flag from the origial command if you want the container to remain intact upon exit.

## License
The licenses for this project can be found inside the LICENSE file. Please take a look at them before proceeding further.
