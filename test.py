import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
import matplotlib.pyplot as plt
import math, time
from math import sqrt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import itertools
import datetime
from operator import itemgetter
import urllib.request, json
import datetime as dt

# List all files under the input directory from kaggle
import os

api_key = random.choice(['F62MHL3VDUMCEMGP', '7DAKX5A7OSW2STDL', 'ELNT38TEYXQOL7WS', 'RU0OE9SIH6R38HXY', 'IXYARDFBT1Y30V7J', 'VGMETC1M5S3ME3AH', 'XG4BIITOYH3PF5GG', 'SBBSF3RLWBOPQ6E6'])

    # American Airlines stock market prices
ticker = "TSLA"

# JSON file with all the stock market data for AAL from the last 20 years
url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(ticker,api_key)

# Save data to this file
file_to_save = './stock_market_data-%s.csv'%ticker

# If you haven't already saved data,
# Go ahead and grab the data from the url
# And store date, low, high, volume, close, open values to a Pandas DataFrame
if not os.path.exists(file_to_save):
    with urllib.request.urlopen(url_string) as url:
        print("Response from Alpha Vantage:", url.getcode())
        data = json.loads(url.read().decode())
        print("Data fetched from Alpha Vantage:", data)
        # extract stock market data
        data = data['Time Series (Daily)']
        df = pd.DataFrame(columns=['Date','Low','High','Close','Open'])
        for k,v in data.items():
            date = dt.datetime.strptime(k, '%Y-%m-%d')
            data_row = [date.date(),float(v['3. low']),float(v['2. high']),
                        float(v['4. close']),float(v['1. open'])]
            df.loc[-1,:] = data_row
            df.index = df.index + 1
    print('Data saved to : %s'%file_to_save)        
    df.to_csv(file_to_save)
