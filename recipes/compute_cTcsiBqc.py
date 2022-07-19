# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
training_data = dataiku.Dataset("training_data")
training_data_df = training_data.get_dataframe()




# Write recipe outputs
MLflow_model = dataiku.Folder("cTcsiBqc")
MLflow_model_info = MLflow_model.get_info()
