# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
eval_data_ = dataiku.Dataset("eval_data_")
df = eval_data_.get_dataframe()

y = df['y']
X = df.drop('y', axis=1)

job_df = pd.get_dummies(df.job, prefix='job')
marital_df = pd.get_dummies(df.marital, prefix='marital')
education_df = pd.get_dummies(df.education, prefix='education')

month_dic = {
    'apr': 5,
    'aug': 8,
    'dec': 12,
    'feb': 2,
    'jan': 1,
    'jul': 7,
    'jun': 6,
    'mar': 3,
    'may': 5,
    'nov': 11,
    'oct': 10,
    'sep': 9}

X.drop(['job', 'marital', 'education'], axis=1, inplace=True)
X = pd.concat([X, job_df, marital_df, education_df], axis=1)
X.month.replace(month_dic, inplace=True)
X.replace({"no": 0, "yes": 1}, inplace=True)

# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

eval_data_prepared_df = pd.concat([X, y], axis=1) # For this sample code, simply copy input to output


# Write recipe outputs
eval_data_prepared = dataiku.Dataset("eval_data_prepared")
eval_data_prepared.write_with_schema(eval_data_prepared_df)