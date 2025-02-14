#!/usr/bin/env python
# coding: utf-8

# ***
# 
# # __Utilizing Machine Learning for High Frequency Algorithmic Trading__
# 
# 
# ##### __Name:__ Masixole Boya<br>__Student number:__ 1869204
# 
# ***

# # __Imports__

# In[1]:


from datetime import datetime
import pandas as pd
import pandas_ta as ta
import seaborn as sns
import dask.dataframe as dd
import numpy as np
import pandas as pd
from scipy.stats import linregress
import json
from scipy.stats import linregress
# import MetaTrader5 as mt5
import mplfinance as mpf
import warnings
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import tensorflow 
from tensorflow.keras import regularizers
from tensorflow import keras
from keras.models import Sequential
from keras.metrics import RootMeanSquaredError
from tensorflow.keras import layers
from keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import pytz
import wandb
# from wandb.keras import WandbCallback, WandbMetricsLogger
from wandb.integration.keras import WandbCallback,WandbMetricsLogger

# from secret_login import Secret_Login, IC_Markets_Login, Wandb_Login
import optuna
from optuna.integration import TFKerasPruningCallback
from tensorflow.keras.models import load_model
import sys

# sys.path.append('../../login_info')
sys.path.append('../../modules')


# Load the credentials from the JSON file
# with open('../../login_info/login_info.json', 'r') as f:
#     credentials = json.load(f)

# import MT5_IC_secret_login
# from MT5_IC_secret_login import Secret_Login, IC_Markets_Login, Wandb_Login, OctaFx_Login



# from MT5_IC_login_function import login_to_mt5
# from mt5_ic_login_function import login_to_mt5



# ### __Imports to be disregarded on remote__

# In[2]:


# import MetaTrader5 as mt5
# from mt5_ic_login_function import login_to_mt5


# # __1. MetaTrader 5 API__ : Setup

# ### __1.1 Initialize Application__

# In[3]:


# mt5.initialize()


# ### __1.2 Login__
# 

# In[4]:


# login_success = login_to_mt5()


# # __2. MetaTrader 5 API__ : Retrieving Data

# In[5]:


# ticker = 'EURUSD'
# frequency = mt5.TIMEFRAME_M1

# # Define the date range correctly
# to_date = datetime.now()
# from_date = datetime(2024, 5, 5)


# In[6]:


# range_rates = mt5.copy_rates_range(ticker, frequency, from_date, to_date)

# range_rates


# In[7]:


# data = pd.DataFrame(range_rates)
# data


# ### ------------------ save data file, delete if exists ---------------------------------

# In[8]:


# data.to_csv('range_rates.csv', index=False)


# ### ------------------- move data file to remote ---------------------------------

# scp range_rates.csv mboya@146.141.21.100:/home-mscluster/mboya/capstone_project_remote/Capstone_Project/trainedModel_historical/MetaTrader5

# # --------------- __START RUNNING ON REMOTE__ --------------

# In[9]:


data = pd.read_csv("range_rates.csv")
data.head(5)


# In[10]:


# data['time'] = pd.to_datetime(data['time'],unit = 's')


# In[11]:


# print(f'first time : \n{data.head(1)['time']}')
# print(f"\nLast time : \n{data.tail(1)['time'] }")


# # __3. Preprocessing Data__

# ### __3.1 Data Cleaning__ 

# In[12]:


def clean_data(data):
    
    # 1. Remove duplicate rows
    print("------- Removing Duplicate Rows -------")
    duplicates = data[data.duplicated()]
    if not duplicates.empty:
        print("Duplicate rows found and removed:\n", duplicates)
    else:
        print("No duplicate rows found.")
    data = data.drop_duplicates()
    
    # 2. Handle missing values
    print("\n------- Handling Missing Values -------")
    missing_values = data[data.isnull().any(axis=1)]
    if not missing_values.empty:
        print("Rows with missing values found and handled:\n", missing_values)
    else:
        print("No missing values found.")
    data = data.dropna()

    # 3. Convert data types if necessary
    print("\n------- Converting Data Types -------")
    if 'time' in data.columns:
        # data['time'] = pd.to_datetime(data['time'], unit='s')
        data['time'] = pd.to_datetime(data['time'],unit = 's')
        print("'time' column converted to datetime.")

     # 4. Remove specific columns: tick_volume, spread, real_volume
    print("\n------- Removing non-OHLC Columns -------")
    columns_to_remove = ['tick_volume', 'spread', 'real_volume']
    existing_columns = [col for col in columns_to_remove if col in data.columns]
    if existing_columns:
        print(f"Removing columns: {existing_columns}")
        data = data.drop(columns=existing_columns)
    else:
        print("No columns to remove.")

    # 4. Remove any rows where OHLC values are 0
    print("\n------- Removing Rows with Zero OHLC Values -------")
    ohlc_columns = ['open', 'high', 'low', 'close']
    zero_ohlc = data[(data[ohlc_columns] == 0).any(axis=1)]
    if not zero_ohlc.empty:
        print("Rows with zero OHLC values found and removed:\n", zero_ohlc)
    else:
        print("No rows with zero OHLC values found.")
    data = data[(data[ohlc_columns] != 0).all(axis=1)]
    
    # 5. (Optional) Handle outliers
    # print("\n------- Handling Outliers (if any) -------")
    # This is where you could implement outlier handling and print details.
    # Example: using Z-score method, etc.
    
    print("\nData cleaning complete.")
    return data


# In[13]:


data = clean_data(data)
data


# ### __3.2 Exploratory Data Analysis__

# In[14]:


def full_eda(data):
    
    warnings.filterwarnings('ignore')
    # 1. Data Information
    print("------- Data Information --------\n")
    print(data.info())
    
    # 2. Data Description
    print("\n---------- Data Description ----------\n")
    print(pd.DataFrame(data.describe()))
    
    # 3. Pairplot for All Columns
    print("\n---------- Pairplot ----------")
    sns.pairplot(data[['open', 'high', 'low', 'close']], diag_kind='kde', plot_kws={'alpha': 0.5})
    plt.figure(figsize=(5, 5))
    plt.show()
    
    # # 6. Line Plot for Close Price
    # print("\n---------- Line Plot for Close Price ----------")
    # plt.figure(figsize=(10, 5))
    # plt.plot(data['close'], label='Close Price')
    # plt.title('Close Price Over Time')
    # plt.xlabel('Date')
    # plt.ylabel('Close Price')
    # plt.legend()
    # plt.show()
    
    # 4. Convert 'time' to datetime and Set Index
    data['time'] = pd.to_datetime(data['time'], unit='s')  # Convert Unix timestamp to datetime
    data.set_index('time', inplace=True)
    
    # 5. Candlestick Chart
    print("\n---------- Candlestick Chart ----------")
    last_hour_data = data.last('1H')
    if not last_hour_data.empty:
        mpf.plot(last_hour_data, type='candle', style='charles', title='Candlestick Chart for the Last Hour', ylabel='Price')
    else:
        print("No data available for the last hour.")
    
    # mpf.plot(data, type='candle', style='charles', title='Candlestick Chart', ylabel='Price')
    
    

    # 7. Histograms and KDEs for OHLC Data
    print("\n---------- Distribution of OHLC ----------")
    plt.figure(figsize=(10, 5))
    columns = ['open', 'high', 'low', 'close']
    for i, col in enumerate(columns):
        plt.subplot(2, 2, i + 1)  # 2x2 grid for 4 plots
        sns.histplot(data[col], kde=True, bins=30, color=sns.color_palette("tab10")[i])
        plt.title(f'{col.capitalize()} Distribution')
    plt.tight_layout()
    plt.show()

    # 8. Box Plots for OHLC Data
    print("\n---------- Box Plot ----------")
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=data[columns])
    plt.title('Box Plots for OHLC Data')
    plt.xlabel('OHLC')
    plt.ylabel('Value')
    plt.show()

    # # # 9. Multiply Values by 100,000 and Plot Again
    # # print("\n---------- Multiplying OHLC values by 100,000 and Plotting Again ----------")
    # # data_scaled = data[columns] * 100000
    # # plt.figure(figsize=(14, 10))
    # # for i, col in enumerate(columns):
    # #     plt.subplot(2, 2, i + 1)  # 2x2 grid for 4 plots
    # #     sns.histplot(data_scaled[col], kde=True, bins=30, color=sns.color_palette("tab10")[i])
    # #     plt.title(f'{col.capitalize()} Distribution (Scaled)')
    # # plt.tight_layout()
    # # plt.show()

    
    # New Section: Plot Prices for Each Minute by Day of the Week for the First Week
    
    print("\n---------- Daily Variations by minute  ----------")
    # Filter data for August 2024 based on the time column
    august_data = data[(data.index.month == 8) & (data.index.year == 2024)].copy()  # Make a copy for August 2024 data

    # Create a column for 'minute_of_day' (time in minutes from 00:00)
    august_data['minute_of_day'] = august_data.index.hour * 60 + august_data.index.minute

    # Get unique days in August from the time column
    unique_days_in_august = august_data.index.normalize().unique()

    # Plot each day's closing prices
    plt.figure(figsize=(8, 4))

    for day in unique_days_in_august:
        day_data = august_data[august_data.index.normalize() == day]
        
        # Only plot if the day has data
        if not day_data.empty:
            plt.plot(day_data['minute_of_day'], day_data['close'], label=day.strftime('%Y-%m-%d'), alpha=0.7)

    # Set plot titles and labels
    plt.title('Closing Prices for Each Day of August 2024 by Minute of the Day')
    plt.xlabel('Minute of the Day')
    plt.ylabel('Close Price')

    # Set x-ticks to represent each hour (0 to 1440 minutes for 24 hours)
    plt.xticks(ticks=range(0, 1441, 60), labels=[f"{i // 60:02d}:{i % 60:02d}" for i in range(0, 1441, 60)], rotation=45)
    plt.xlim(0, 1440)  # 0 to 1440 minutes (24 hours)

    # Add grid, legend, and adjust layout
    plt.grid()
    plt.legend(title='Day of August', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
        


 # --- August 2024 Dates Based on the Calendar ---
    august_dates = {
        'Monday': ['2024-08-05', '2024-08-12', '2024-08-19', '2024-08-26'],
        'Tuesday': ['2024-08-06', '2024-08-13', '2024-08-20', '2024-08-27'],
        'Wednesday': ['2024-08-07', '2024-08-14', '2024-08-21', '2024-08-28'],
        'Thursday': ['2024-08-01', '2024-08-08', '2024-08-15', '2024-08-22', '2024-08-29'],
        'Friday': ['2024-08-02', '2024-08-09', '2024-08-16', '2024-08-23', '2024-08-30'],
        'Saturday': ['2024-08-03', '2024-08-10', '2024-08-17', '2024-08-24', '2024-08-31'],
        'Sunday': ['2024-08-04', '2024-08-11', '2024-08-18', '2024-08-25']
    }

    # Create a column for 'minute_of_day' (time in minutes from 00:00) in a temporary dataframe
    temp_data = data.copy()  # Use a copy of the original dataframe
    temp_data['minute_of_day'] = temp_data.index.hour * 60 + temp_data.index.minute

    # --- Section 1: Plot Combined Days of Week (All Mondays, All Tuesdays, etc.) ---
    print("\n---------- Section 1: Combined Weekdays ----------")

    plt.figure(figsize=(8, 4))
    for day, dates in august_dates.items():
        day_data = temp_data[temp_data.index.normalize().isin(pd.to_datetime(dates))]  # Use the temporary dataframe
        avg_day_data = day_data.groupby('minute_of_day')['close'].mean()
        plt.plot(avg_day_data.index, avg_day_data.values, label=day, alpha=0.7)

    plt.title('Combined Days: Average Close Price for Each Day of Week (August 2024)')
    plt.xlabel('Minute of the Day')
    plt.ylabel('Average Close Price')
    plt.xticks(ticks=range(0, 1441, 60), labels=[f"{i // 60:02d}:{i % 60:02d}" for i in range(0, 1441, 60)], rotation=45)  # Set x-ticks for each hour
    plt.xlim(0, 1440)  # 0 to 1440 minutes (24 hours)
    plt.grid()
    plt.legend(title='Day of the Week')
    plt.tight_layout()
    plt.show()

    # --- Section 2: Individual Weekdays with Lines for Each Weekday ---
    print("\n---------- Section 2: Individual Weekdays ----------")

    for day, dates in august_dates.items():
        plt.figure(figsize=(8, 4))
        for date in dates:
            specific_day_data = temp_data[temp_data.index.normalize() == pd.to_datetime(date)]  # Use the temporary dataframe
            if not specific_day_data.empty:
                plt.plot(specific_day_data['minute_of_day'], specific_day_data['close'], label=date, alpha=0.7)
        
    plt.title(f'Close Prices for Each Minute of the Day - {day}s (August 2024)')
    plt.xlabel('Minute of the Day')
    plt.ylabel('Close Price')
    plt.xticks(ticks=range(0, 1441, 60), labels=[f"{i // 60:02d}:{i % 60:02d}" for i in range(0, 1441, 60)], rotation=45)
    plt.xlim(0, 1440)
    plt.grid()
    plt.legend(title=f'Each {day}', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()




# In[15]:


full_eda(data)


# # __4. Technical Analysis__

# ## __4.1 Feature Enginnering__ (Indicators)

# In[16]:


data.ta.indicators()


# #### __4.1.1 Simple Moving Average (SMA)__

# In[17]:


data['SMA_10'] = ta.sma(data['close'], length=10)
# data['SMA_20'] = ta.sma(data['close'], length=20)


# #### __4.1.2 Exponential Moving Average (EMA)__

# In[18]:


data['EMA_10'] = ta.ema(data['close'], length=10)


# #### __4.1.3 Moving Average Convergence Divergence (MACD)__

# In[19]:


macd = ta.macd(data['close'])

# Add the MACD, Signal, and Histogram to your DataFrame
# data['MACD'] = macd['MACD_12_26_9']
# data['MACD_Signal'] = macd['MACDs_12_26_9']
data['MACD_Histogram'] = macd['MACDh_12_26_9']


# #### __4.1.4 Relative Strength Index (RSI)__

# In[20]:


data['RSI_14'] = ta.rsi(data['close'], length=14)


# #### __4.1.6 Williams %R__

# In[21]:


data['Williams_%R'] = ta.willr(data['high'], data['low'], data['close'], length=14)


# In[22]:


data.sample(6)


# ## 4.2 __Feature Engineering__ (Time Variations)
# ### __4.2.1 Minute of the Day__

# In[23]:


# Create the 'minute_of_day' feature as an integer from 1 to 1440 (based on the time index)
data['minute_of_day'] = data.index.hour * 60 + data.index.minute + 1


# ### __4.2.2 Hour of the Day__

# In[24]:


# Create the 'hour_of_day' feature as an integer from 1 to 24 (based on the time index)
data['hour_of_day'] = data.index.hour + 1


# ## __4.3 Featrue Engineering__ (Calculating Slope)
# 
# I calculate the slope of the indicators over a rolling window using linear regression. The slope tells us the direction ( or indicates the rate of change in the values over time) and strength of the trend over a specified number of periods.
# 
# The calculated slope reflects the trend's direction and magnitude:
# 
# - Positive Slope: Indicates an upward trend in the data.
# - Negative Slope: Indicates a downward trend.
# - Slope Magnitude: The greater the magnitude, the steeper the trend.

# In[25]:


def get_slope(array):
    '''
    Parameters
    array: A sequence of numerical values (e.g., SMA values) over which the slope will be calculated. The function expects this to be a NumPy array.
    Returns
    slope: The slope of the linear regression line fitted to the input array.
    '''
    y = np.array(array)
    x = np.arange(len(y))
    slope, intercept, r_value, p_value, std_err = linregress(x,y)
    return slope


# In[26]:


data.sample(5)


# # __5. Preparing Data__

# ### __5.1 Scaling__

# In[27]:


# Create a new column 'log_return' based on the 'close' price
data['log_return'] = np.log(data['close'] / data['close'].shift(1))

# Remove any rows where 'log_return' is NaN (occurs at the start of the data due to shift)
data.dropna(subset=['log_return'], inplace=True)


# #### re-arranging the dataset for ease of comprehension

# In[28]:


# Drop the original 'close' column
data.drop(columns=['close'],inplace= True)

# Rename 'log_return' to 'close'
data.rename(columns={'log_return': 'close'},inplace=True)


# In[29]:


# Now, rearrange the columns
rearranged_columns = [
    'hour_of_day',
    'minute_of_day',
    'close', 
    'SMA_10',
    'EMA_10',
    'MACD_Histogram',
    'RSI_14',
    'Williams_%R'
]

# Create a new DataFrame with the rearranged columns
data = data[rearranged_columns]

data


# ### __5.2 Creating training data__
# 
# #### __5.2.1 Preparing Lagged Data__
# 
# - __`df`:__ This is a DataFrame containing the original data.
#   
# - __`lag_steps:`__ This is an integer representing the number of time steps to lag the time unit by. It determines how many previous values of the target variable ('close') to include in the lagged DataFrame.
#   
# - __`lagged_df:`__ This is a new DataFrame that will store the lagged data.
#   
# - __`pd.DataFrame(index=df.index)`:__ This creates a new DataFrame with the same index as the original DataFrame df. The index represents the date and time of each data point.
#   
# - __`lagged_df['DateTime'] = df.index:`__ This creates a new column in the lagged_df DataFrame called 'DateTime', and it copies the index of the original DataFrame df. This column will store the date and time information.
#   
# - __`lagged_df['ActualValue'] = df['close']:`__ This creates a new column in the lagged_df DataFrame called 'ActualValue', and it copies the values from the 'close' column of the original DataFrame df. This column will store the actual values of the target variable.
#   
# - __`Loop:`__ This loop iterates from 1 to lag_steps (inclusive). For each iteration:
#     - __`lagged_df[f'PrevValue_{i}'] = df['close'].shift(i):`__ This creates a new column in the lagged_df DataFrame for each lagged value. The column name includes the prefix 'PrevValue_' followed by the lag index i. It shifts the values of the 'close' column of the original DataFrame df upwards by i time steps and stores them in the new column. This effectively creates lagged features for the target variable.
#   
# - __`return lagged_df.dropna():`__ This returns the lagged DataFrame after dropping any rows with missing values (NaN). Since creating lagged features involves shifting the data, the first few rows will contain NaN values where there is no data available for the lagged features.
# 

# In[30]:


data.columns


# In[ ]:


# go_back_by = 3

# def create_lagged_dataframe(df, lag_steps=go_back_by, columns_to_lag=None):
#     if columns_to_lag is None:
#         columns_to_lag = df.columns  # By default, lag all columns

#     lagged_df = pd.DataFrame(index=df.index)
#     lagged_df['DateTime'] = df.index
#     lagged_df['ActualValue'] = df['close']  # Keep 'ActualValue' for the close price

#     # Creating lagged columns for each selected column
#     for column in columns_to_lag:
#         for i in range(1, lag_steps + 1):
#             lagged_df[f'{column}_lag_{i}'] = df[column].shift(i)

#     return lagged_df.dropna()

# # Specify the columns you want to lag
# columns_to_lag = ['hour_of_day','minute_of_day','close', 'SMA_10','EMA_10' ,'RSI_14','Williams_%R', 'MACD_Histogram']

# lagged_df = create_lagged_dataframe(data, lag_steps=go_back_by, columns_to_lag=columns_to_lag)


go_back_by = 3

def create_lagged_dataframe(df, lag_steps=go_back_by, columns_to_lag=None):
    if columns_to_lag is None:
        columns_to_lag = df.columns  # By default, lag all columns

    lagged_df = pd.DataFrame(index=df.index)
    lagged_df['DateTime'] = df.index
    lagged_df['ActualValue'] = df['close']  # Keep 'ActualValue' for the close price

    # Creating lagged columns in reverse order for each selected column
    for column in columns_to_lag:
        for i in range(lag_steps, 0, -1):  # Start from lag_steps, go down to 1
            lagged_df[f'{column}_lag_{i}'] = df[column].shift(i)

    return lagged_df.dropna()

# Specify the columns you want to lag
columns_to_lag = ['hour_of_day','minute_of_day','close', 'SMA_10','EMA_10' ,'RSI_14','Williams_%R', 'MACD_Histogram']

lagged_df = create_lagged_dataframe(data, lag_steps=go_back_by, columns_to_lag=columns_to_lag)



# In[ ]:


lagged_df


# In[33]:


lagged_df.shape


# In[34]:


lagged_df.columns


# #### __5.2.2 Prepare LSTM Data__
# 
# - __`lagged_df:`__ This is a DataFrame containing the lagged data.
# 
# - __`dates:`__ This variable stores the date and time information from the lagged DataFrame.
# 
# - __`X:`__ This variable stores the input features for the LSTM model. It consists of all columns from the lagged DataFrame except 'DateTime' and 'ActualValue', converted to float32 data type.
# 
# - __`y:`__ This variable stores the target variable for the LSTM model, which is the 'ActualValue' column from the lagged DataFrame, converted to float32 data type.
# 
# - __`return:`__ The function returns three variables: dates, X, and y, containing the respective data.
# 
# 

# In[35]:


def prepare_lstm_data(lagged_df):
    dates = lagged_df['DateTime']
    X = lagged_df.drop(columns=['DateTime', 'ActualValue']).astype(np.float32)
    y = lagged_df['ActualValue'].astype(np.float32)

    return dates, X.values, y.values



dates, X, y = prepare_lstm_data(lagged_df)
print("Dates:", dates.shape)
print("X shape:", X.shape)
print("y shape:", y.shape)


# ### __5.3 Spliting: Training, Validation, Testing Data__

# #### __Split Data__
# 
# - __`dates:`__ This variable contains the date and time information.
# 
# - __`X:`__ This variable contains the input features for the model.
# 
# - __`y:`__ This variable contains the target variable for the model.
# 
# - __`x_train:`__ This variable contains the input features for the training set.
# 
# - __`y_train:`__ This variable contains the target variable for the training set.
# 
# - __`x_val:`__ This variable contains the input features for the validation set.
# 
# - __`y_val:`__ This variable contains the target variable for the validation set.
# 
# - __`x_test:`__ This variable contains the input features for the test set.
# 
# - __`y_test:`__ This variable contains the target variable for the test set.
# 
# - __`train_data:`__ This tuple contains the input features and target variable for the training set.
# 
# - __`val_data:`__ This tuple contains the input features and target variable for the validation set.
# 
# - __`test_data:`__ This tuple contains the input features and target variable for the test set.
# 
# - __`return:`__ The function returns three tuples: train_data, val_data, and test_data, each containing the respective input features and target variable for the corresponding set.
# 
# 

# In[36]:


def split_data(dates, X, y):
    x_train, x_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

(train_data, val_data, test_data) = split_data(dates, X, y)
(x_train, y_train), (x_val, y_val), (x_test, y_test) = train_data, val_data, test_data


# In[37]:


x_train_view = pd.DataFrame(x_train)
x_train_view


# In[38]:


y_tran_view = pd.DataFrame(y_train)
y_tran_view


# In[39]:


# OPTIONAL: Calculating the total number of instances
total_instances = len(x_train) + len(x_val) + len(x_test)

# Each set has
print("Training set:", len(x_train), "instances (", len(x_train) / total_instances * 100, "%)")
print("Validation set:", len(x_val), "instances (", len(x_val) / total_instances * 100, "%)")
print("Testing set:", len(x_test), "instances (", len(x_test) / total_instances * 100, "%)")


# 
# # __6. Training the model__

# ### __6.1 Weights and Biases__

# In[40]:


# !wandb login 92747ba767c68ec2ec63d2b44818eaeba8e973b9

import subprocess

# # Log in to WandB
subprocess.run(['wandb', 'login', '92747ba767c68ec2ec63d2b44818eaeba8e973b9'])


# ### __6.1.1 Defining the model__

# In[41]:


# Define the custom Directional Accuracy (DA) metric
def directional_accuracy(y_true, y_pred):
    # Compute the true differences between consecutive time steps
    diff_true = y_true[1:] - y_true[:-1]
    
    # Compute the predicted differences between consecutive time steps
    diff_pred = y_pred[1:] - y_pred[:-1]
    
    # Check if the directions (signs) match and calculate directional accuracy
    correct_directions = tensorflow.reduce_sum(tensorflow.cast((diff_true * diff_pred) > 0, tensorflow.float32))
    
    # Divide by the number of comparisons made (n-1 due to the time difference)
    da = correct_directions / tensorflow.cast(tensorflow.shape(diff_true)[0], tensorflow.float32)
    
    # Return DA as a percentage
    return da * 100.0

# run = wandb.init(
#     project='first_keras_intergration',
#     config={
#         'learning_rate':1.5057750726359656e-05,
#         'epochs' : 10,
#         'batch_size': 64,
#         'loss_function' : 'mean_squared_error',
#         'architecture' : 'LSTM',
#         }
#     )



# config =  wandb.config

# tensorflow.keras.backend.clear_session()

# #the actaul neural network
# wb_model = Sequential([
#     layers.Input((go_back_by,1)),

#     layers.LSTM(10, return_sequences=True),
#     layers.Dropout(0.2), 
#     layers.LSTM(5), 
#     layers.Dense(4, activation='relu', kernel_regularizer='l2'),  
#     layers.Dropout(0.2),  
#     layers.Dense(4, activation='relu', kernel_regularizer='l2'), 
     
#     layers.Dense(1)
# ])

# wb_model.summary()

# # compile
# wb_model.compile(
#     loss = config.loss_function, 
#     optimizer =  Adam(learning_rate=config.learning_rate),
#     # metrics = ['mean_absolute_error']
#     metrics=['mean_absolute_error', 'mse', RootMeanSquaredError(), directional_accuracy]
#     )


# In[ ]:


# Initialize WandB for tracking the training process
run = wandb.init(
    project='first_keras_integration',
    config={
        'learning_rate': 1.5057750726359656e-03,
        'epochs': 3,
        'batch_size': 64,  # Default batch size, can be adjusted
        'loss_function': 'mean_squared_error',
        'architecture': 'LSTM',
        'dropout_rate': 0.14153675379172143
    }
)

# Use WandB config
config = wandb.config

# Clear any previous TensorFlow session
tensorflow.keras.backend.clear_session()

# # Build the LSTM model using the given hyperparameters
# wb_model_tuned = Sequential([
#     layers.Input((go_back_by, 1)),

#     layers.LSTM(59, return_sequences=True),
#     layers.Dropout(0.14153675379172143),
    
#     layers.LSTM(16, return_sequences=True),
#     layers.Dropout(0.14153675379172143),
    
#     layers.LSTM(62, return_sequences=True),
#     layers.Dropout(0.14153675379172143),
    
#     layers.LSTM(35, return_sequences=True),
#     layers.Dropout(0.14153675379172143),
    
#     layers.LSTM(37),  # Last LSTM layer without return_sequences
#     layers.Dropout(0.14153675379172143),
    
#     # Add dense layers as per the architecture
#     layers.Dense(4, activation='relu', kernel_regularizer='l2'),
#     layers.Dropout(0.14153675379172143),
#     layers.Dense(4, activation='relu', kernel_regularizer='l2'),

#     # Output layer (single output for regression)
#     layers.Dense(1)
# ])

wb_model_tuned = Sequential([
    layers.Input((go_back_by, 1)),
    layers.LSTM(20, return_sequences=True),
    layers.Dropout(0.1),
    layers.LSTM(8),
    layers.Dropout(0.1),
    layers.Dense(2, activation='tanh', kernel_regularizer=regularizers.l2(0.02)),
    # layers.Dropout(0.2),
    layers.Dense(1)
])

# Print the model summary (optional)
wb_model_tuned.summary()

# Compile the model
# wb_model_tuned.compile(
#     loss=config.loss_function,
#     optimizer=Adam(learning_rate=config.learning_rate),
#     metrics=['mean_absolute_error', 'mean_squared_error', RootMeanSquaredError(), directional_accuracy]
# )

# compile
wb_model_tuned.compile(
    loss = config.loss_function, 
    optimizer =  Adam(learning_rate=config.learning_rate),
    # metrics = ['mean_absolute_error']
    metrics=['mean_absolute_error', 'mse', RootMeanSquaredError(), directional_accuracy]
    )


# ### __6.1.2 Training the model__

# In[42]:


wandb_history = wb_model_tuned.fit(
    x_train,y_train, 
    epochs = config.epochs,
    batch_size =  config.batch_size,
    validation_data = (x_test, y_test),
    callbacks = [WandbMetricsLogger()]
    )


# In[44]:


# # loss, mae, mse, rmse = wb_model.evaluate(x_test, y_test, verbose=0)
# # Adjust the unpacking to account for all metrics including custom metrics
# loss, mae, mse, rmse, da = wb_model.evaluate(x_test, y_test, verbose=0)


# #test data
# predictions = wb_model.predict(x_test)

# # Calculate RMSE and R-squared
# rmse_manual = np.sqrt(mean_squared_error(y_test, predictions))
# r2 = r2_score(y_test, predictions)

# # Print evaluation metrics
# print('\nTest Loss:', loss)
# print('\nMean Absolute Error:', mae)
# print('\nRoot Mean Squared Error (from Keras):', rmse)
# print('\nRoot Mean Squared Error (manual calculation):', rmse_manual)
# print('\nR-squared:', r2)
# print(f"Directional Accuracy: {da}")


# In[44]:


# wb_model.save("second_real_model.keras")


# # __7. Hyperparameter Tuning__

# In[45]:


# def objective(trial):
#     num_layers = trial.suggest_int('num_layers', 1, 5)
#     layer_units = [trial.suggest_int(f'layer_units_{i}', 4, 64) for i in range(num_layers)]
#     activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'sigmoid'])
#     epochs = trial.suggest_int('epochs', 2, 10)
#     dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
#     learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    
#     # Initialize WandB run
#     # wandb.init(project='first_keras_intergration', config={
#     #     'learning_rate': learning_rate,
#     #     'epochs': epochs,
#     #     'batch_size': 64,
#     #     'loss_function': 'mean_squared_error',
#     #     'architecture': 'LSTM',
#     #     'dropout_rate': dropout_rate
#     # })
    
#     config = {
#         'learning_rate': learning_rate,
#         'epochs': epochs,
#         'batch_size': 64,
#         'loss_function': 'mean_squared_error',
#         'architecture': 'LSTM',
#         'dropout_rate': dropout_rate
#     }

#     # Clear previous session
#     tensorflow.keras.backend.clear_session()

#     # Create the Keras model
#     wb_model = Sequential([
#         layers.Input((go_back_by, 1)),
#         layers.LSTM(layer_units[0], return_sequences=True),
#         layers.Dropout(dropout_rate)
#     ])

#     for units in layer_units[1:-1]:
#         wb_model.add(layers.LSTM(units, return_sequences=True))
#         wb_model.add(layers.Dropout(dropout_rate))
    
#     wb_model.add(layers.LSTM(layer_units[-1], return_sequences=False))
#     wb_model.add(layers.Dropout(dropout_rate))
#     wb_model.add(layers.Dense(4, activation=activation, kernel_regularizer='l2'))
#     wb_model.add(layers.Dropout(dropout_rate))
#     wb_model.add(layers.Dense(4, activation=activation, kernel_regularizer='l2'))
#     wb_model.add(layers.Dense(1))

#     # Compile the model
#     wb_model.compile(
#         loss=config['loss_function'],
#         optimizer=Adam(learning_rate=config['learning_rate']),
#         metrics=['mean_absolute_error', 'mean_squared_error', RootMeanSquaredError()]
#     )
    
#     # Train the model
#     wandb_history = wb_model.fit(
#         x_train, y_train,
#         epochs=config['epochs'],
#         batch_size=config['batch_size'],
#         validation_data=(x_test, y_test)
#         # No WandB callbacks here
#     )
    
#     # Evaluate the model
#     val_loss = wandb_history.history['val_loss'][-1]
#     return val_loss


# study = optuna.create_study(
#     storage="sqlite:///db.sqlite3",
#     direction='minimize', 
#     study_name='keras_lstm_opt_1'
#     )
# study.optimize(
#     objective, 
#     n_trials=5
#     )


# In[46]:


# best_trial = study.best_trial

# print("Best hyperparameters: ", best_trial.params)


# In[47]:


# best_trial = study.best_trial

# # Extract the best hyperparameters
# best_params = best_trial.params

# # Print the best hyperparameters
# print("Best hyperparameters: ", best_params)

# # # Save the best hyperparameters to a .txt file
# # with open("best_hyperparameters.txt", "w") as file:
# #     file.write("Best hyperparameters:\n")
# #     for key, value in best_params.items():
# #         file.write(f"{key}: {value}\n")

# print("Best hyperparameters:\n")
# for key, value in best_params.items():
#     print(f"{key}: {value}\n")


# In[48]:


# import tensorflow.keras.backend as K
# K.clear_session()


# ### __7.2 Training tuned model__

# In[ ]:


# import tensorflow as tf
# from tensorflow.keras import layers, Sequential
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.metrics import RootMeanSquaredError
# import wandb

# # Initialize WandB for tracking the training process
# run = wandb.init(
#     project='first_keras_integration',
#     config={
#         'learning_rate': 1.5057750726359656e-05,
#         'epochs': 10,
#         'batch_size': 64,  # Default batch size, can be adjusted
#         'loss_function': 'mean_squared_error',
#         'architecture': 'LSTM',
#         'dropout_rate': 0.14153675379172143
#     }
# )

# # Use WandB config
# config = wandb.config

# # Clear any previous TensorFlow session
# tf.keras.backend.clear_session()

# # Build the LSTM model using the given hyperparameters
# wb_model_tuned = Sequential()

# # Add the input layer
# wb_model_tuned.add(layers.Input((go_back_by, 1)))

# # Add 5 LSTM layers with specified units and dropout layers
# wb_model_tuned.add(layers.LSTM(59, return_sequences=True))
# wb_model_tuned.add(layers.Dropout(0.14153675379172143))

# wb_model_tuned.add(layers.LSTM(16, return_sequences=True))
# wb_model_tuned.add(layers.Dropout(0.14153675379172143))

# wb_model_tuned.add(layers.LSTM(62, return_sequences=True))
# wb_model_tuned.add(layers.Dropout(0.14153675379172143))

# wb_model_tuned.add(layers.LSTM(35, return_sequences=True))
# wb_model_tuned.add(layers.Dropout(0.14153675379172143))

# wb_model_tuned.add(layers.LSTM(37))  # Last LSTM layer without return_sequences
# wb_model_tuned.add(layers.Dropout(0.14153675379172143))

# # Add dense layers as per the architecture
# wb_model_tuned.add(layers.Dense(4, activation='relu', kernel_regularizer='l2'))
# wb_model_tuned.add(layers.Dropout(0.14153675379172143))
# wb_model_tuned.add(layers.Dense(4, activation='relu', kernel_regularizer='l2'))

# # Output layer (single output for regression)
# wb_model_tuned.add(layers.Dense(1))

# # Print the model summary (optional)
# wb_model_tuned.summary()

# # Compile the model
# wb_model_tuned.compile(
#     loss=config.loss_function,
#     optimizer=Adam(learning_rate=1.5057750726359656e-05),
#     metrics=['mean_absolute_error', 'mean_squared_error', RootMeanSquaredError(), directional_accuracy]
# )

# # Train the model with WandB logging
# wb_model_tuned.fit(
#     x_train, y_train,
#     epochs=10,  # Hardcoded value for epochs
#     batch_size=64,  # Hardcoded batch size
#     validation_data=(x_test, y_test),
#     callbacks=[wandb.keras.WandbCallback()]  # Ensure WandB logs the training process
# )

# # Finish the WandB run
# run.finish()


# In[ ]:


wb_model_tuned.save("second_real_model_TUNED_3.0.keras")


# # __8. Evaluation (Tuned model)__

# In[ ]:


loss, mae, mse, rmse, da = wb_model_tuned.evaluate(x_test, y_test, verbose=0)


#test data
predictions = wb_model_tuned.predict(x_test)

# Calculate RMSE and R-squared
rmse_manual = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

# Print evaluation metrics
print('\nTest Loss:', loss)
print('\nMean Absolute Error:', mae)
print('\nRoot Mean Squared Error (from Keras):', rmse)
print('\nRoot Mean Squared Error (manual calculation):', rmse_manual)
print('\nR-squared:', r2)
print(f"Directional Accuracy: {da}")

# wb_model_tuned.save("LSTM_model_TESTING.keras")


# In[ ]:


# plt.figure(figsize=(10, 6))
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

