# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import os
import shutil
import dataiku
import pandas as pd
import mlflow
from sklearn.linear_model import LogisticRegression
from dss_mlflow import preprocessing

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
client = dataiku.api_client()
project = client.get_default_project()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# get train dataset
train_dataset = dataiku.Dataset("training_data")

# get train dataset as a pandas dataframe
df = train_dataset.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# get output saved model
sm = project.get_saved_model("VdHxdbkg")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# get the path of a local managed folder where to temporarily save the trained model
mf = dataiku.Folder("cTcsiBqc")
path = mf.get_path()

model_subdir = "my_subdir"
model_dir = os.path.join(path, model_subdir)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
X, y = preprocessing.clean_df(df)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
if os.path.exists(model_dir):
    shutil.rmtree(model_dir)

try:
    # ...train your model...
    clf = LogisticRegression(random_state=0).fit(X, y)

    # ...save it with package specific MLflow method (here, SKlearn)...
    mlflow.sklearn.save_model(clf, model_dir)

    # import the model, creating a new version
    mlflow_version = sm.import_mlflow_version_from_managed_folder("v03", "cTcsiBqc", model_subdir, "py36_mlflow")
finally:
    shutil.rmtree(model_dir)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# setting metadata (target name, classes,...)
mlflow_version.set_core_metadata(target_column_name="y",
                             class_labels=["no", "yes"],
                             get_features_from_dataset="eval_data_prepared")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# evaluate the performance of this new version, to populate the performance screens of the saved model version in DSS
mlflow_version.evaluate("eval_data_prepared")