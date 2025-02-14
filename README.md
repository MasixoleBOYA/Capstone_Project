# Capstone Project in Data Science / Artificial Intelligence
#### ` Artificial Intelligence for Retail traders in High Frequency Algorithmic Trading`

## Beginner's Intoduction to Forex Trading
#### The following recommended material is for complete beginners to learn about the basics of FOREX trading

#### - `Trading, Investment and Portfolio Management`: [click here](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Trading%2C+Investment+and+Portfolio+Management+BY%3A+ERIK+PAULSON+VISHAL+K.+RATHI+HTAY+AUNG+WIN+Submitted%3A+February+2017&btnG=)
   - PAULSON, E., RATHI, V.K., WIN, H.A., HAKIM, H. and RADZICKI, M.J., 2017. Trading, Investment and Portfolio Management (Doctoral dissertation, WORCESTER POLYTECHNIC INSTITUTE).

#### - `Forex Basics Course`
   - Octa Broker. (2020). Forex Basics Course. Available at: [click here](https://www.youtube.com/playlist?list=PLwi9xUIQFHIwzGRYwdLpf35aKA29Zm3jW)

## To go to the training file:
- Go to __trainedModel_historical__ folder, then __MetaTrader5__, then __mt5_model.ipynb__ or __mt5_model.py__.  
- Run the code in one of the following 3 ways:
  1. Run the python file, which outputs on the terminal.
  2. To use a notebook, run the code cell that starts after the `Start running from here on remote` section on the notebook.
  3. Alternatively, satisfy the following conditions in order to call the data API:
    - The `MT5 Terminal` desktop application must be installed
    - Register with the `MC Markets` broker and login on the MT5 Terminal with your details
    - Intergrate the code with your credentials.
   

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

        - *mt5_model.ipynb*: __Main file__ Contains all model training code.
        - *.keras*: are saved models.
        - *.txt*: are saved outputs.     
    - `Tiingo/`
      - Trained model(s) using the Tiingo API.
    - `yfinance/`
      - Trained model(s) using the yfinance API.


