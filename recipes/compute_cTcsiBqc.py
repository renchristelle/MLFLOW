# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import os
import shutil
import dataiku

from dataiku import recipe

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
client = dataiku.api_client()
project = client.get_default_project()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# get train dataset
train_dataset = recipe.get_inputs_as_datasets()[0]
evaluation_dataset = recipe.get_inputs_as_datasets()[1]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# get output saved model
sm = project.get_saved_model(recipe.get_output_names()[0])

# get train dataset as a pandas dataframe
df = train_dataset.get_dataframe()

# get the path of a local managed folder where to temporarily save the trained model
mf = dataiku.Folder("local_managed_folder")
path = mf.get_path()

model_subdir = "my_subdir"
model_dir = os.path.join(path, model_subdir)

if os.path.exists(model_dir):
    shutil.rmtree(model_dir)

try:
    # ...train your model...

    # ...save it with package specific MLflow method (here, SKlearn)...
    mlflow.sklearn.save_model(my_model, model_dir)

    # import the model, creating a new version
    mlflow_version = sm.import_mlflow_version_from_managed_folder("version_name", "local_managed_folder", model_subdir, "code-env-with-mlflow-name")
finally:
    shutil.rmtree(model_dir)

# setting metadata (target name, classes,...)
mlflow_version.set_core_metadata(target_column, ["class0", "class1",...] , get_features_from_dataset=evaluation_dataset.name)

# evaluate the performance of this new version, to populate the performance screens of the saved model version in DSS
mlflow_version.evaluate(evaluation_dataset.name)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
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