# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns

# def boxplot_plotting (num_rows,num_columns,width,height,variables,datafr,number):
#     """
#     A function which returns a given number of boxplots for different target  against each numerical feature. The returning objects are seaborn.boxplot types. 
    
#     -------------------
#     PARAMETERS:
#     A dataframe containing the variables and their correspondent labels
#     Variables: A list of each variable's name
#     num_rows and num_columns: An integer and positive number for both num_rows and num_columns for the
#     boxplot fig "canvas" object where our boxplots will go,
#     width: A positive width measure 
#     length: A positive length measure 
#     A binary class label 
#     A column array for managing variable names
#     A training dataframe object
#     Integer positive number for correct ordering  of graphs 
#     -------------------
#     REQUISITES:
#     The target labels ("class label") must be within the data frame 
#     The multiplication between num_rows and num_columns must return be equal to num_variables.
#     It is possible for num_rows & num_columns to be values that when multiplied don't equal the "variables" numeric value,
#     but that will create more boxplots which will be empty. 
    

#     --------------------
#     RETURNS:
#     It returns a fixed number "num_variables" of boxplot objects. Each Boxplot represents both Target Class
#     Labels according to a given Variable

#     --------------------
#     Examples

#     datafr=train_df
#     --------
#     boxplot_plotting (3,3,20,25,numeric_column,datafr,number)
#     """
#     fig,ax= plt.subplots(num_rows,num_columns,figsize=(width,height))
#     for idx, (var,subplot) in enumerate(zip(variables,ax.flatten())):
#         a = sns.boxplot(x='class',y=var,data=datafr,ax=subplot).set_title(f"Figure {number}.{idx}: Boxplot of {var} for each target class label")
#     return fig