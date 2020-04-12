# sk-ml-pipeline
End-to-end sklearn model training, bayesian hyperparameter tuning, experiment tracking, and deployment using MLflow.

**The current version of the package works for classification problems and the 
[Breast Cancer Wisconsin](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data) dataset. 
If you want to train models, the data must be downloaded in the `local_data_path` specified in the config file.**

## Helpers
Docker is used for the environment management, it has to be installed in your machine. The following scripts
have been created to automatize the most important pipeline steps:
- Build the container: `bin/build`
- Train a model: `bin/train`
- Run the container: `bin/run`
- Execute the tests: `bin/tests`
- Deploy a model: `bin/deploy {experiment_id} {run_id}`. The experiment and run identifiers are generated automatically
by MLflow.
