CeFlow: A Robust and Efficient Counterfactual Explanation Framework for Tabular Data using Normalizing Flows
==============================

Overview
CeFlow is a robust and efficient counterfactual explanation framework for tabular data that uses normalizing flows. The framework provides interpretable and accurate counterfactual explanations for individual predictions of machine learning models. This README provides an overview of the repository structure and how to use it.

Repository Structure
The repository contains the following folders and files:

data/: This folder contains the data used in the experiments. The data folder includes subfolders train and test, which contain the training and testing datasets respectively.

reports/: This folder contains the report(s) generated from the experiments, such as the evaluation metrics and visualizations.

src/: This folder contains the source code for the CeFlow framework. It includes the implementation of normalizing flows and counterfactual explanations for tabular data.

train_ce_flow.py: This script is used to train the CeFlow model on the training data.

train_flow.py: This script is used to train the normalizing flow model on the training data.

run_gs.py: This script is used to run the grid search to find the best hyperparameters for the CeFlow model.

Usage
To use the CeFlow framework, follow these steps:

Clone the repository to your local machine.
Navigate to the src/ folder.
Run train_flow.py to train the normalizing flow model on the training data.
Run train_ce_flow.py to train the CeFlow model on the training data.
Run run_gs.py to perform a grid search to find the best hyperparameters for the CeFlow model.
After training the model, run predict.py to obtain counterfactual explanations for individual predictions.
The reports/ folder will contain the evaluation metrics and visualizations generated from the experiments.

Requirements
The CeFlow framework requires the following Python packages:

PyTorch
NumPy
Pandas
Scikit-learn
Matplotlib
Seaborn
These packages can be installed using pip.

Conclusion
The CeFlow framework provides a robust and efficient counterfactual explanation framework for tabular data using normalizing flows. The implementation includes training scripts, grid search, and prediction scripts to obtain counterfactual explanations for individual predictions. The framework can be extended and modified to suit different applications and datasets.
