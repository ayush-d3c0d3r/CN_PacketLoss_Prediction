# CN_PacketLoss_Prediction
CN_LabProject
 Ayush Panda- 22052893
 Harsh sankrit- 22051075
 Priyanshu shekhar- 22051445
 Divyansh Bajpai- 22052980
 Suraj Das- 22053207
 Overview
 The CN_LabProject implements Support Vector Regression (SVR) to predict packet
 loss rates based on a dataset containing various network-related features. Packet loss
 is a critical metric in network performance that impacts user experience, especially in
 real-time applications. This project aims to provide a predictive model that can
 estimate packet loss rates effectively.
 Objectives
  DataPreprocessing: Load and prepare the data for analysis, including handling
 any necessary transformations.
  ModelTraining: Train an SVR model using the training dataset to understand
 the relationship between network features and packet loss.
  ModelEvaluation: Evaluate the model's performance using key metrics such as
 Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² Score.
  Visualization: Use Seaborn to create informative plots that visualize the
 performance metrics.
 Installation
 To run this project, ensure you have Python installed along with the following
 libraries:
  pandas:For data manipulation and analysis.
  scikit-learn: For implementing the SVR model and evaluation metrics.
  matplotlib: For plotting visualizations.
  seaborn: For enhanced data visualization.
 You can install the required libraries using pip:
 ```bash
 pip install pandas scikit-learn matplotlib seaborn
 The dataset used in this project is expected to be in CSV format and named
 modified_train.csv. It should include the following:
 Features: Various network-related metrics (e.g., latency, bandwidth) that could
 influence the packet loss rate.
Target Variable: Packet Loss Rate indicating the percentage of packets lost during
 transmission.
 Usage
 Load the Dataset: The first step is to load the dataset into a Pandas DataFrame. The
 code reads the CSV file into memory.
 Data Preprocessing:Drop the target variable (Packet Loss Rate) from the feature set.
 Split the dataset into training and testing sets (80% training, 20% testing) using
 train_test_split from scikit-learn.
 Feature Scaling: Standardize the feature values using StandardScaler. This ensures
 that the features contribute equally to the distance calculations used by SVR.
 Model Training: Initialize the SVR model with a linear kernel and fit it to the scaled
 training data.
 Making Predictions: Use the trained model to predict packet loss rates on the scaled
 test set.
 Model Evaluation:
 Calculate performance metrics: MSE, MAE, and R² Score to assess how well the
 model performs.
 Visualizations:Create bar plots for each metric using Seaborn to visualize the model's
 performance clearly.
 Example Workflow
 Runthe Script: After ensuring that the modified_train.csv file is present in the same
 directory, execute the Python script in a Jupyter Notebook or another Python
 environment.
 Check Results: After running the script, the console will display the calculated
 metrics:
 Mean Squared Error (MSE)
 Mean Absolute Error (MAE)
 R² Score
 Visualize Metrics: The script will also generate plots visualizing the evaluation
 metrics, providing insights into the model's performance.
 Evaluation Metrics
 MeanSquaredError (MSE): Represents the average of the squared differences
 between predicted and actual values. Lower values indicate a better fit.
  MeanAbsoluteError (MAE): Indicates the average absolute error between
 predicted and actual values. Similar to MSE, lower values indicate better model
 performance.
  R²Score: This score represents the proportion of variance in the dependent
 variable explained by the independent variables. Values close to 1 indicate a good
 fit, while negative values suggest that the model does not explain the variance
 well.
 Visualizations
 The visualizations include separate bar plots for each evaluation metric:
 Mean Squared Error (MSE) Bar Plot: Shows the magnitude of the MSE value.
 Mean Absolute Error (MAE) Bar Plot: Represents the MAE value clearly.
 R² Score Bar Plot: Displays the R² value, illustrating how well the model explains the
 variance in packet loss.
 These visualizations help in quickly assessing the model's performance and
 identifying areas for improvement.
 Conclusion
 This project effectively demonstrates the application of SVR for predicting packet
 loss rates in network data. The evaluation metrics and visualizations provide insight
 into the model's performance, highlighting its strengths and weaknesses. Future work
 may involve experimenting with different kernels or algorithms to improve accuracy
 and reliability.
 Acknowledgements:
 This project was conducted as part of a lab exercise to explore machine learning
 techniques in the context of network performance metrics.
