# DSCI-310-Group-5

## Team member:
- Tien Nguyen
- Claudio Torres Cantú
- Edison Le


## Summary
The present project seeks to provide a prediction to the problematic of spotting clear patterns of benign and malignant tumors, which comes from the question "Is there a way to efficiently classify whether a tumor is malignant or benign with high accuracy,  given a set of different features observed from the tumor in its development stage, using the support of the  computing technologies currently available?". 
Such problematic was resolved using a predictive model. Our initial hypothesis was that it could be possible to do so yet it would be very inefficient and would have a high error rate. Another method of thinking and problem approaching was necessary.
After performing EDA, like summary statistics and data cleaning and visualization, 
we tested multiple different classification models and arrived at a K-Nearest-Neighbor model with tuned hyperparameters with very good accuracy, recall, precision and f1 score. 

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

In order to run the code,advise you to use the Dockerfile listed in the main branch to have a suitable environment and avoid any problems with dependencies. The code is intended to run in JupyterLab.

using the command:

`docker pull nhantien/dsci310group5:<tagname>`

where `<tagname>` is the version of the image.

Another option would be to clone the remote repository to your local one, and run the command:

`docker run -it nhantien/dsci310group5:<tagname>`

or if you want to remove the container after exit and keep the image:

`docker run -it --rm nhantien/dsci310group5:<tagname>`

Once you have activated the new environment, you can type "Jupyter lab" in the command line which will take you to the web version of jupyter notebook. Once there you should see the repository you copied. 

Finally, the license we are using for the current project is the MIT License. It is briefly described in Github.com as "A short and simple permissive license with conditions only requiring preservation of copyright and license notices. Licensed works, modifications, and larger works may be distributed under different terms and without source code" 
Further information can be found at:
https://opensource.org/licenses/MIT


