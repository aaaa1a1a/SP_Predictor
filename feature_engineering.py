'''
- May be interesting: Interest rates: https://quant.stackexchange.com/questions/20911/how-to-predict-daily-range-of-forex
Feature transformation:
- For every S&P day make Close = next Open
- Generate Y: BUY, SELL, HOLD at time t signals if p(t + 1) is higher, lower or the same as p(t), lets call it "Signal"
- Put together the S&P OHLC, S&P vol and closing prices of all the potential features in one file
- Normalize the data: Calculate daily log returns on the price data
- Put all of the above in one file

Feature selection:
- Choose the top 4 indeces using:
- Perform correlation matrix analysis
- PCA
- Make a call in case of conflict between the above 2 methods i.e. pick what we think is better


'''

import pandas as pd
import download_data as dd
import numpy as np


class Signals:
    def HOLD(self): return int(0)

    def BUY(self): return int(1)

    def SELL(self): return int(2)

    def SIGNAL(self):
        return "Signal"


SIGNALS = Signals()


# def equalize_close_open()

def generate_y(df, col_name):
    diff = df[col_name].diff(periods=-1)
    diff.values[diff.values == 0] = SIGNALS.HOLD()
    diff.values[diff.values > 0] = SIGNALS.SELL()
    diff.values[diff.values < 0] = SIGNALS.BUY()
    return diff


def log_returns(df, col_name):
    ratio = df[col_name] / df[col_name].shift(1)
    return np.log(ratio)


def standardize(df, col_name):
    col = df[col_name]
    mean = col.mean()
    std = col.std()
    return ((col - mean) / std)


# def harmonize_dates():
SP_file_name = dd.file_name("SP", dd.interval_period)
SP = pd.read_csv(SP_file_name)
print(standardize(SP, "Volume"))
# SP[SIGNALS.SIGNAL()] = generate_y(SP, "Close")
# SP.to_csv(SP_file_name)
