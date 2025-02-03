# Improving Heart Disease Prediction Accuracy: An Exploration of Logistic Regression and KNN

Developers: Sahiel Bose & Shanay Gaitonde

Research Article: https://medium.com/nextgen-innovations/improving-heart-disease-prediction-accuracy-an-exploration-of-logistic-regression-and-knn-5e4af2aed66c

HeartKNN is a machine learning-based system designed to classify heart disease risk using K-Nearest Neighbors (KNN) and Linear Regression. By leveraging data-driven techniques, the model provides accurate predictions to aid early detection and prevention of heart disease. The project is built using Python and machine learning libraries such as scikit-learn, offering a scalable and efficient solution for medical data analysis.

Project Highlights
The project focuses on enhancing model robustness through preprocessing, implementing KNN and Linear Regression for classification, and evaluating performance using accuracy, precision, recall, and a confusion matrix. It is scalable and customizable, allowing for fine-tuning with different datasets and parameters.

Preprocessing Pipeline
The preprocessing stage includes feature scaling with normalization to ensure data consistency. Missing values are handled using imputation techniques, and important features are selected based on correlation and importance scores.

Model Architecture
The project uses K-Nearest Neighbors (KNN) for classification, which is a distance-based algorithm, and Linear Regression for trend analysis to understand contributing factors. KNN hyperparameters, such as the number of neighbors, are fine-tuned to optimize performance.

Training Strategy
An adaptive optimizer is used to improve model convergence, and the loss function is evaluated using classification metrics. Cross-validation ensures the model generalizes well and prevents overfitting.

Post-Training Evaluation
After training, the model is tested on a separate dataset, with key evaluation metrics including accuracy, precision, recall, and the confusion matrix. These metrics provide insights into the model’s classification performance.

Dataset
The dataset includes health-related attributes such as age, cholesterol levels, and blood pressure. It is split into a training set for learning, a validation set for hyperparameter fine-tuning, and a test set for final performance assessment.

Results
The model achieves high accuracy after hyperparameter tuning. Visualization tools, such as training curves and the confusion matrix, help analyze the model's effectiveness.
