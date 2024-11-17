# Capstone Project in Data Science / Artificial Intelligence
#### ` Artificial Intelligence for Retail traders in High Frequency Algorithmic Trading`

## Beginner's Intoduction to Forex Trading
#### The following recommended material is for complete beginners to learn about the basics of FOREX trading

#### - `Trading, Investment and Portfolio Management`: [click here](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Trading%2C+Investment+and+Portfolio+Management+BY%3A+ERIK+PAULSON+VISHAL+K.+RATHI+HTAY+AUNG+WIN+Submitted%3A+February+2017&btnG=)
   - PAULSON, E., RATHI, V.K., WIN, H.A., HAKIM, H. and RADZICKI, M.J., 2017. Trading, Investment and Portfolio Management (Doctoral dissertation, WORCESTER POLYTECHNIC INSTITUTE).

#### - `Forex Basics Course`
   - Octa Broker. (2020). Forex Basics Course. Available at: [click here](https://www.youtube.com/playlist?list=PLwi9xUIQFHIwzGRYwdLpf35aKA29Zm3jW)



## Folder Structure

- `indicators/`
  - This folder contains the files for the calculated indicators. More detailed information can be found in the folder.

- `login_info/`
  - This folder contains login credentials for the demo trading accounts. *Real trading accounts' login credentials will be hosted in a separate private repository, for security, which will then be linked to this repository.*

- `real_time_WebSocket/`
  - This folder contains files for pulling real-time data using WebSockets in Python with the Tiingo API. More detailed information can be found in the folder.

- `trainedModel_historical/`
  - This folder contains files for training models with historical data using various data sources. The MetaTrader5 is the main folder.:
    - `MetaTrader5/`
      - Main folder with the models used finally. Trained model(s) using the MT5 API.
      - __Main file:__ 
        - *mt5_model.ipynb*: Contains all model training code.
      - __Other files:__ 
        - the *.keras* are saved models.
        - the *.txt* are saved outputs.     
    - `Tiingo/`
      - Trained model(s) using the Tiingo API.
    - `yfinance/`
      - Trained model(s) using the yfinance API.
