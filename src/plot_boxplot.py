import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def boxplot_plotting (num_rows,num_columns,width,height,variables,datafr,number):
    """
    A function which returns a given number of boxplots for different target against     each numerical feature. The returning objects are seaborn.boxplot types. 
    
    -------------------
    PARAMETERS:
	datafr:
			A dataframe containing the variables and their correspondent labels
	variables:
			A list of string values corresponding to each variable's name (name of columns)
	num_rows:
			An integer and non negative number which corresponds to the number of rows for the boxplot canvas object where the boxplots (subplots) will be displayed.
	num_columns: 
				An integer and non negative number which corresponds to the number of columns for the boxplot canvas object where the boxplots (subplots) will be displayed.
	Width: A positive numerical measure for the width of each of the subplots (boxplots)
	Length: A positive numerical measure for the length of each of the subplots

    -------------------
    REQUISITES:
	The target labels ("class label") must be within the data frame 
	The class label must be binary
	The multiplication between num_rows and num_columns must return be equal to num_variables.
	It is possible for num_rows & num_columns to be values that when multiplied don't equal the "variables" numeric value,
	but that will create more boxplots which will be empty. 
    

    --------------------
    RETURNS:
    It returns a fixed number "num_variables" of boxplot objects. Each Boxplot represents both Target Class
    Labels according to a given Variable

    --------------------
    Examples

    datafr=train_df
    --------
    boxplot_plotting (3,3,20,25,numeric_column,datafr,number)
    """
    fig,ax= plt.subplots(num_rows,num_columns,figsize=(width,height))
    for idx, (var,subplot) in enumerate(zip(variables,ax.flatten())):
        a = sns.boxplot(x='class',y=var,data=datafr,ax=subplot, hue="class").set_title(f"Figure {number}.{idx+1}: Boxplot of {var} for each target class label")
    return fig