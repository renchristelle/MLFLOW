# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import os
import shutil
import dataiku
import pandas as pd
import mlflow

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
client = dataiku.api_client()
project = client.get_default_project()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# get train dataset
train_dataset = dataiku.Dataset("training_data")
evaluation_dataset = dataiku.Dataset("eval_data")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# get output saved model
sm = project.get_saved_model("VdHxdbkg")

# get train dataset as a pandas dataframe
df = train_dataset.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# get the path of a local managed folder where to temporarily save the trained model
mf = dataiku.Folder("cTcsiBqc")
path = mf.get_path()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
model_subdir = "my_subdir"
model_dir = os.path.join(path, model_subdir)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from sklearn.linear_model import LogisticRegression

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
y = df['y']
X = df.drop('y', axis=1)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
job_df = pd.get_dummies(df.job, prefix='job')
marital_df = pd.get_dummies(df.marital, prefix='marital')
education_df = pd.get_dummies(df.education, prefix='education')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
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

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
X.drop(['job', 'marital', 'education'], axis=1, inplace=True)
X = pd.concat([X, job_df, marital_df, education_df], axis=1)
X.month.replace(month_dic, inplace=True)
X.replace({"no": 0, "yes": 1}, inplace=True)
clf = LogisticRegression(random_state=0).fit(X, y)

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