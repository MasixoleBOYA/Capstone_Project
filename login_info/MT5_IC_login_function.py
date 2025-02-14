
import MetaTrader5 as mt5
import json

# Load the credentials from the JSON file
with open('../../login_info/login_info.json', 'r') as f:
    credentials = json.load(f)

def login_to_mt5():
    """
    Logs into the MetaTrader5 platform using the IC Markets demo account credentials.
    
    Returns:
        bool: True if the login is successful, False otherwise.
    """
    # Print current account info
    print(mt5.account_info().login)
    print(mt5.account_info().server)
    
    # Retrieve login credentials from the JSON data
    ic_markets_login = credentials["IC_Markets_Login"]
    account = ic_markets_login['login']
    password = ic_markets_login['password']
    server = ic_markets_login['server']
    
    # print(f"Account : {account}")
    # print(f"Password : {password}")
    # print(f"Server : {server}")
    
    # Attempt to log in
    authorized = mt5.login(account, password, server)
    
    if authorized:
        print(f"Connected to account #{account}")
        return True
    else:
        print(f"Failed to connect to account #{account}, error code: {mt5.last_error()}")
        return False
