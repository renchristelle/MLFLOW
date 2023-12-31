# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
from dss_mlflow import preprocessing

# Read recipe inputs
eval_data_ = dataiku.Dataset("eval_data_")
df = eval_data_.get_dataframe()

X, y = preprocessing.clean_df(df)

# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

eval_data_prepared_df = pd.concat([X, y], axis=1) # For this sample code, simply copy input to output


# Write recipe outputs
eval_data_prepared = dataiku.Dataset("eval_data_prepared")
eval_data_prepared.write_with_schema(eval_data_prepared_df)