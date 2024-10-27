# CN_LabProject

**Authors:**  
Ayush Panda - 22052893  
Harsh Sankrit - 22051075  
Priyanshu Shekhar - 22051445  
Divyansh Bajpai - 22052980  
Suraj Das - 22053207  

## Overview
The **CN_LabProject** implements Support Vector Regression (SVR) to predict packet loss rates based on a dataset containing various network-related features. Packet loss is a critical metric in network performance that significantly impacts user experience, especially in real-time applications. This project aims to provide a predictive model that effectively estimates packet loss rates.

## Objectives
- **Data Preprocessing:** Load and prepare the data for analysis, including handling necessary transformations.
- **Model Training:** Train an SVR model using the training dataset to understand the relationship between network features and packet loss.
- **Model Evaluation:** Evaluate the model's performance using key metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² Score.
- **Visualization:** Use Seaborn to create informative plots that visualize the performance metrics.

## Installation
To run this project, ensure you have Python installed along with the following libraries:
- **pandas**: For data manipulation and analysis.
- **scikit-learn**: For implementing the SVR model and evaluation metrics.
- **matplotlib**: For plotting visualizations.
- **seaborn**: For enhanced data visualization.

You can install the required libraries using pip:
pip install pandas scikit-learn matplotlib seaborn
## Usage
1. **Load the Dataset:** Load the dataset into a Pandas DataFrame.
2. **Data Preprocessing:**
   - Drop the target variable (Packet Loss Rate) from the feature set.
   - Split the dataset into training and testing sets (80% training, 20% testing) using `train_test_split` from scikit-learn.
3. **Feature Scaling:** Standardize the feature values using `StandardScaler`. This ensures that the features contribute equally to the distance calculations used by SVR.
4. **Model Training:** Initialize the SVR model with a linear kernel and fit it to the scaled training data.
5. **Making Predictions:** Use the trained model to predict packet loss rates on the scaled test set.
6. **Model Evaluation:**
   - Calculate performance metrics: MSE, MAE, and R² Score to assess model performance.
7. **Visualizations:** Create bar plots for each metric using Seaborn to visualize the model's performance clearly.

## Example Workflow
1. **Run the Script:** Ensure that the `modified_train.csv` file is present in the same directory, then execute the Python script in a Jupyter Notebook or another Python environment.
2. **Check Results:** The console will display the calculated metrics:
   - Mean Squared Error (MSE)
   - Mean Absolute Error (MAE)
   - R² Score
3. **Visualize Metrics:** The script will also generate plots visualizing the evaluation metrics, providing insights into the model's performance.

## Evaluation Metrics
- **Mean Squared Error (MSE):** Represents the average of the squared differences between predicted and actual values. Lower values indicate a better fit.
- **Mean Absolute Error (MAE):** Indicates the average absolute error between predicted and actual values. Similar to MSE, lower values indicate better model performance.
- **R² Score:** Represents the proportion of variance in the dependent variable explained by the independent variables. Values close to 1 indicate a good fit, while negative values suggest that the model does not explain the variance well.

## Visualizations
The visualizations include separate bar plots for each evaluation metric:
- **Mean Squared Error (MSE) Bar Plot:** Shows the magnitude of the MSE value.
- **Mean Absolute Error (MAE) Bar Plot:** Represents the MAE value clearly.
- **R² Score Bar Plot:** Displays the R² value, illustrating how well the model explains the variance in packet loss.

These visualizations help in quickly assessing the model's performance and identifying areas for improvement.
```bash

## Conclusion
This project effectively demonstrates the application of SVR for predicting packet loss rates in network data. The evaluation metrics and visualizations provide insight into the model's performance, highlighting its strengths and weaknesses. Future work may involve experimenting with different kernels or algorithms to improve accuracy and reliability.

## Acknowledgements
This project was conducted as part of a lab exercise to explore machine learning techniques in the context of network performance metrics.
