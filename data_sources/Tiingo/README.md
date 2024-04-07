

## Project Overview
This project aims to train a machine learning model using intraday financial data retrieved from the Tiingo API. The model is trained to predict future prices based on historical data. The project involves several key steps, including data retrieval, analysis, preprocessing, model training, and evaluation.

## Libraries Used
- pandas: For data manipulation and analysis.
- requests: For making HTTP requests to the Tiingo API.
- matplotlib: For data visualization.
- mplfinance: For plotting financial charts.
- scikit-learn: For data preprocessing and evaluation.
- numpy: For numerical computing.
- keras and tensorflow: For building and training the LSTM model.
- datetime and pytz: For handling date and time information.

## Steps Involved
1. **Requesting the data**: Data is retrieved from the Tiingo API, including both intraday data and top-of-book data.
2. **Analysis & Processing**:
    - Basic analysis is performed on the retrieved data, including checking shape and info.
    - Data is preprocessed by scaling using StandardScaler and creating lagged features for LSTM training.
    - Data is split into training, validation, and testing sets.
3. **Training the model**: A Sequential LSTM model is defined and compiled, then trained on the training data.
4. **Model Evaluation**: The trained model is evaluated using the testing data, and metrics such as loss, mean absolute error, root mean squared error, and R-squared are calculated.

## Instructions
1. Clone this repository to your local machine.
2. Install the required libraries using pip or conda.
3. Ensure you have obtained an API token from Tiingo and replace it in the code.
4. Run the provided Python script, following the instructions and comments provided within.
5. Analyze the results and tweak parameters as necessary to improve model performance.

## Note
- Make sure to adhere to Tiingo's API usage guidelines and terms of service while accessing the data.
- Experiment with different hyperparameters, architectures, and preprocessing techniques to optimize model performance.
- Feel free to customize the code or extend the project with additional features or analyses.

For more detailed information on each step and code snippets, refer to the sections above.
