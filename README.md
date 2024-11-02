# NVIDIA Stock Price Prediction #

This project predicts NVIDIA's stock prices based on historical stock data. Using a combination of moving averages, trading volume, and closing prices, the project applies machine learning models to forecast future prices. This type of project is a valuable addition to a data science portfolio, demonstrating time series analysis, feature engineering, and predictive modeling skills.


## Introduction ##
The objective of this project is to predict stock prices using historical stock data for NVIDIA. The project explores multiple machine learning techniques for time series forecasting, with the aim of finding the model that best predicts future prices.

## Dataset ##
The dataset contains daily stock data for NVIDIA, including:

- Open, High, Low, Close Prices: Daily price metrics
- Volume: Number of shares traded
- Adj Close: Adjusted closing price, accounting for factors like dividends

## Project Structure ##
The project consists of the following steps:

1. Data Preprocessing: Handle missing values, create moving averages, and scale features.
2. Exploratory Data Analysis (EDA): Visualize price trends, moving averages, and trading volume.
3. Modeling: Apply models like Linear Regression to predict stock prices.
4. Evaluation: Use error metrics such as Mean Absolute Error (MAE) and Mean Squared Error (MSE) to evaluate model performance.
5. Prediction Visualization: Plot predicted vs. actual prices for visual assessment.

## Data Preprocessing ##
- Feature Engineering: Added 5-day and 20-day moving averages of the closing price.
- Scaling: Applied feature scaling for better model performance.
- Data Splitting: Divided data into training (80%) and testing (20%) sets.

## Modeling## 
The project employs a Linear Regression model as a starting point. Additional models, such as Random Forest or Long Short-Term Memory (LSTM) networks, can be explored in future iterations for improved accuracy.

## Evaluation ##
Model performance is evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
  
These metrics provide insights into the accuracy of predictions and help in model tuning.

## Results ##
The results indicate that the model can capture general trends in NVIDIA’s stock price but may require further tuning for enhanced precision. Visualizations of predicted vs. actual prices reveal the model's accuracy over the testing period.

## Future Work ##
Potential improvements include:

- Experimenting with advanced time-series models (e.g., ARIMA, LSTM).
- Adding external factors, like macroeconomic indicators, to improve predictive power.
- Hyperparameter tuning to optimize model performance.

## Requirements ##

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- scikit-learn

## Usage ##
To run the project:

1. Preprocess the data: Execute the code in the Data Preprocessing section.
2. Train and test the model: Use the code in the Modeling section to train and make predictions.
3. Evaluate and visualize results: Run the Evaluation and Prediction Visualization sections.
