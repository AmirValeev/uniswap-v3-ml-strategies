import pandas as pd


def calculateRSI(prices_data, n=14, today_price=None):
    """Calculate the Relative Strength Index of an asset.
    Args:
        prices_data (pandas dataframe object): prices data
        n (int, optional): number of . Defaults to 14.
        today_price(int, optional): today's price to predict future RSI. Defaults to None
    Return:
        rsi (pandas series object): relative strength index
    """
    price = prices_data['quotePrice']

    # Append today's date if used for prediction
    if today_price is not None:
        price = price.append(pd.Series({price.size: today_price}))

    delta = price.diff()
    delta = delta[1:]

    prices_up = delta.copy()
    prices_up[prices_up < 0] = 0
    prices_down = delta.copy()
    prices_down[prices_down > 0] = 0

    roll_up = prices_up.rolling(n).mean()
    roll_down = prices_down.abs().rolling(n).mean()

    relative_strength = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))

    return rsi


def calculateMACD(prices_data):
    """Calculate the MACD of EMA15 and EMA30 of an asset
    Args:
        prices_data (dataframe): prices data
    Returns:
        macd (pandas series object): macd of the asset
        macd_signal (pandas series object): macd signal of the asset
    """
    ema15 = pd.Series(prices_data['quotePrice'].ewm(
        span=15, min_periods=15).mean())
    ema30 = pd.Series(prices_data['quotePrice'].ewm(
        span=30, min_periods=30).mean())

    macd = pd.Series(ema15 - ema30)
    macd_signal = pd.Series(macd.ewm(span=9, min_periods=9).mean())

    return macd, macd_signal


def extractAll(data):
    """Generate most important technical indicators for an asset
    Including -
    EMA9 - exponential moving average for 9 ticks
    SMA5 - simple moving average for 5 ticks
    SMA10 - simple moving average for 10 ticks
    SMA15 - simple moving average for 15 ticks
    SMA30 - simple moving average for 30 ticks
    RSI - Relative Strength Index
    MACD - Moving Average Convergence Divergence
    MACD Signal
    Args:
        data (pandas dataframe object): prices data
    Returns:
        pandas dataframe object: prices data with all the indicators
    """
    prices_data = data.copy()
    # Add moving averages
    prices_data['EMA_9'] = prices_data['quotePrice'].ewm(9).mean().shift()
    prices_data['SMA_5'] = prices_data['quotePrice'].rolling(5).mean().shift()
    prices_data['SMA_10'] = prices_data['quotePrice'].rolling(10).mean().shift()
    prices_data['SMA_15'] = prices_data['quotePrice'].rolling(15).mean().shift()
    prices_data['SMA_30'] = prices_data['quotePrice'].rolling(30).mean().shift()

    # RSI
    # prices_data['RSI'] = calculateRSI(prices_data).fillna(0)

    # MACD
    macd, macd_signal = calculateMACD(prices_data)
    prices_data['MACD'] = macd
    prices_data['MACD_signal'] = macd_signal

    # Shift label(y) by one value to predict the next day using today's data (technical indicators)
    prices_data['quotePrice'] = prices_data['quotePrice'].shift(-1)
    # Drop invalid samples - the samples where moving averages exceed the required window
    prices_data = prices_data.iloc[33:]
    prices_data = prices_data[:-1]  # since we did shifting by one
    # prices_data.index = range(len(prices_data))  # update indexes

    drop_cols = ['baseCurrency', 'quoteCurrency', 'time', 'quoteAmount', 'baseAmount', 'tradeAmount']
    prices_data = prices_data.drop(drop_cols, axis=1)
    return prices_data