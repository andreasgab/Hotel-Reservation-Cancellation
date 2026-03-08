Hotel Booking Cancellation Prediction

This project implements a robust machine learning pipeline to predict hotel booking cancellations. The workflow transitions from initial data exploration and cleaning to advanced hyperparameter tuning and model serialization.

The project was done by Thanos Chronopoulos, Andreas Gabriel & Filippos Georgiopoulos.
Be sure to unzip the pickle files so the streamlit app can run!

Notebook Chapters:

1. Data Processing

    Cleaning: Removes leaky features (e.g., reservation_status) and high-cardinality noise (e.g., company) to ensure model integrity.

    Engineering: Features are processed via a custom smart_encode_hotels function to handle categorical variables.

2. Analysis & Selection

    Feature Selection: Identifies top predictors using SelectKBest and MRMR (Maximum Relevance Minimum Redundancy).

    Interpretability: Employs Partial Dependence Plots to visualize how variables like lead time influence cancellation probability.

3. Model Development

    Benchmarking: Compares 8 baseline models to establish performance standards.

    Optimization: Extensive hyperparameter tuning focused on F1-Score (Balance) and F2-Score (Recall), using XGBoost, CatBoost and RandomForest.

    Evaluation: Uses Confusion Matrices and Cross-Validation to verify stability.

4. Deployment Strategy

    Full Training: Final models are retrained on the entire dataset (X_full​, y_full​) to maximize information gain.

    Serialization: Pipelines are exported as .pickle files, bundling the KNNImputer, StandardScaler, and SelectKBest steps with the final estimator. Parts are also saved independetly.

A full powerpoint is included, showcasing our findings.
Furthermore, we have created a streamlit app that can load new data and run the models on it to produce results and aid in the decision-making.