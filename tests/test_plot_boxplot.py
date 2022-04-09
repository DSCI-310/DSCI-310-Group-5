from unicodedata import numeric
import pytest
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from src.plot_boxplot import boxplot_plotting

test_df = pd.DataFrame({'age':['25','48','30'], 'height': ['185','192','187'],'weight':['85','93','90'], 
'class':['0','1','1'] })
num_df=test_df.apply(pd.to_numeric)
var_names=num_df.head()
number_of_rows=3
number_of_columns=1
width = 50
height = 26
list_example = ["apple", "banana", "cherry"]
test_case = boxplot_plotting(number_of_rows,number_of_columns,width,height,var_names,num_df,3)
b=mpl.figure.Figure()
comparison_var=5

def test_return_type():
    #Test for the correct return type of function:
    assert type(test_case) == type(b)

def test_dataframe_type_values():
    #Tests to see if the values of each column are numeric in order to be able to plot them
    for i in range (len(var_names)):
        assert type(i)==type(comparison_var)
        
def test_integer_values():
    #Test to confirm that num_rows and num_columns are integer values
    assert type(number_of_rows) == type(comparison_var)
    assert type(number_of_rows) == type(comparison_var)
    assert type(width) == type(comparison_var)
    assert type(height) == type(comparison_var)
    

def test_product_consistency():
    #Tests to see if the number of boxplots created will match the number of variables involved. This is 
    #to avoid extra unuseful boxplots or not enough boxplots to show all variables interacting with the class values
    assert number_of_columns * number_of_rows == len(var_names)

def test_wrong_input_dataframe():
    """
    Check TypeError raised when inputting wrong type for what should be a pandas dataframe.
    """
    with pytest.raises(TypeError):
        boxplot_plotting(number_of_rows,number_of_columns,width,height,var_names,list_example,3)
