Below is a ready-to-use README.md drafted from your notebook and assignment brief.

# DA5401 A8 — Ensemble Learning on Bike Sharing

This project builds and compares ensemble regressors to forecast hourly bike rentals (target: cnt) using the UCI Bike Sharing dataset. It implements baselines, Bagging, Gradient Boosting, and a Stacking ensemble, and evaluates all models with RMSE on a chronological holdout split to respect time order.

## Project goals
- Establish baselines: Decision Tree (depth 6) and Linear Regression. 
- Reduce variance via Bagging with shallow trees. 
- Reduce bias via Gradient Boosting. 
- Combine diverse learners with Stacking (KNN, Bagging, Gradient Boosting) and a Ridge meta-learner. 

## Data
- Source: UCI Bike Sharing Demand (hourly) with >17,000 samples. 
- Target: cnt (total rented bikes). 
- Files used: hour.csv (primary), day.csv for context only. 
- Columns dropped as irrelevant or leaky: instant, dteday, casual, registered. 

## Environment
Minimal requirements to run the notebook:
- pandas, numpy, scikit-learn. 

Example requirements.txt:
- pandas
- numpy
- scikit-learn


## Setup and usage
1) Place hour.csv (and optionally day.csv) in the project directory. Refer to the UCI repository for downloads. 
2) Open solution.ipynb and run all cells in order. The notebook handles preprocessing, splitting, modeling, and evaluation. 
## Methodology

### Preprocessing
- Feature groups:
  - Categorical: season, yr, mnth, hr, holiday, weekday, workingday, weathersit. 
  - Numeric: temp, atemp, hum, windspeed. 
- One-hot encode categoricals with handle_unknown="ignore"; passthrough numeric columns via ColumnTransformer. 
- Chronological split (80/20) to avoid future leakage. 

### Models
- Baselines:
  - DecisionTreeRegressor with max_depth=6, min_samples_leaf=10. 
  - Linear Regression. 
- Bagging:
  - BaggingRegressor with DecisionTreeRegressor(base), n_estimators=100, bootstrap sampling. 
- Gradient Boosting:
  - GradientBoostingRegressor with n_estimators=300, learning_rate=0.05, max_depth=3. 
- Stacking:
  - Base learners: KNeighborsRegressor(n_neighbors=10), BaggingRegressor(tree base as above), GradientBoostingRegressor (as above). 
  - Meta-learner: Ridge. 
  - StackingRegressor with cv=5 to generate out-of-fold training predictions for the meta-learner. 

## Results

### Test RMSE (chronological holdout)
- Linear Regression: 133.347. 
- Decision Tree (depth 6, leaf 10): 158.723. 
- Bagging (100× shallow trees): 155.736. 
- Gradient Boosting: 110.447. 
- Stacking (KNN + Bagging + GB, Ridge meta): 103.803. 

Notes:
- Bagging slightly improved over a single shallow tree, consistent with limited variance reduction under high bias. 
- Gradient Boosting substantially reduced error versus baselines, capturing nonlinear interactions. 
- Stacking achieved the lowest RMSE, leveraging model diversity and meta-learning to reduce both bias and variance.

## Discussion

### Why chronological split
Hourly demand is time-dependent; using a chronological 80/20 split prevents leakage from future periods and gives a realistic generalization estimate. [attached_file:file:1]

### Bias–variance reasoning
- Shallow trees with strong regularization have high bias; bagging mainly reduces variance, so gains are modest. 
- Boosting reduces bias by fitting residuals stage-wise with shallow trees, capturing nonlinearities missed by linear and bagged shallow trees.
- Stacking blends diverse learners (local patterns from KNN, variance reduction from bagging, bias reduction from boosting) with a regularized meta-learner (Ridge) trained on out-of-fold predictions, yielding robust improvements. 

## Reproducibility notes
- The notebook uses OneHotEncoder(handle_unknown="ignore") to avoid category-mismatch issues between train and test. 
- RMSE uses root_mean_squared_error where available; older mean_squared_error(..., squared=False) is deprecated in newer sklearn versions. 
- Random states set to 42 for tree-based estimators to stabilize runs. 

## Repository structure
- solution.ipynb: Main analysis, models, and results. 
- A8-Ensembles.pdf: Assignment brief and grading criteria. 
- hour.csv: Dataset file (not included; obtain from UCI). 
- requirements.txt: See Environment section. 

## References
- Dataset: Fanaee-T, Hadi, and Gamper, H. 2014. Bike Sharing Dataset. UCI Machine Learning Repository. 

