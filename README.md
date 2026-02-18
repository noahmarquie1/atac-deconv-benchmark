# ATAC-Seq Deconvolution Pipeline

A systematic benchmark of machine learning methods for bulk ATAC-seq deconvolution. Using DeconPeaker's preprocessing pipeline as a fixed foundation, we evaluate a range of approaches — from regularized regression to gradient boosting and constrained neural networks — to ask how much model complexity chromatin accessibility deconvolution actually requires.

## Methods Tested

1. Non-Negative Least Squares (NNLS)
2. Elastic Net
3. Support Vector Regression (SVR)
4. Gradient Boosting with XGBoost
5. Random Forests
6. ...