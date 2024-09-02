import re
from pandas import DataFrame
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib
from datetime import datetime, timedelta


def extract_minutes(timeframe):
    timescale = timeframe[-1]
    num = int(re.findall('\d+', timeframe)[0])
    if timescale == 'm':
        return num
    elif timescale == 'h':
        return num * 60
    elif timescale == 'd':
        return num * 1440
    elif timescale == 'w':
        return num * 1440 * 7


def heikenashi_smooth(dataframe, timeperiod1: int=10, timeperiod2: int=10) -> DataFrame:
    """
    Calculates the Heikin Ashi Smooth Indicator.
    """
    # Work on a copy of the DataFrame
    bars = dataframe.copy()

    # Calculate the EMA of the original candles
    bars['open'] = ta.EMA(bars['open'], timeperiod=timeperiod1)
    bars['high'] = ta.EMA(bars['high'], timeperiod=timeperiod1)
    bars['low'] = ta.EMA(bars['low'], timeperiod=timeperiod1)
    bars['close'] = ta.EMA(bars['close'], timeperiod=timeperiod1)

    # Calculate the Heikin Ashi candles from the EMA candles
    # --------------------------
    bars['ha_close'] = (bars['open'] + bars['high'] +
                        bars['low'] + bars['close']) / 4
    for i in range(1, len(bars)):
        if (pd.isna(bars.at[i-1, 'open'])) and (not pd.isna(bars.at[i, 'open'])):
            bars.at[i, 'ha_open'] = (bars.at[i, 'open'] + bars.at[i, 'close'])/2
        elif (not pd.isna(bars.at[i-1, 'open'])) and (not pd.isna(bars.at[i, 'open'])):
            bars.at[i, 'ha_open'] = (bars.at[i-1, 'ha_open'] + bars.at[i-1, 'ha_close'])/2
    bars['ha_high'] = bars.loc[:, ['high', 'ha_open', 'ha_close']].max(axis=1)
    bars['ha_low'] = bars.loc[:, ['low', 'ha_open', 'ha_close']].min(axis=1)

    # Smooth out the HA candles one more time
    bars['ha_open_smooth'] = ta.EMA(bars['ha_open'], timeperiod=timeperiod2)
    bars['ha_high_smooth'] = ta.EMA(bars['ha_high'], timeperiod=timeperiod2)
    bars['ha_low_smooth'] = ta.EMA(bars['ha_low'], timeperiod=timeperiod2)
    bars['ha_close_smooth'] = ta.EMA(bars['ha_close'], timeperiod=timeperiod2)

    return bars[
        [
            'ha_open_smooth',
            'ha_high_smooth',
            'ha_low_smooth',
            'ha_close_smooth'
        ]
    ]


def halftrend(dataframe, amplitude=4, channel_deviation=2):
    df = dataframe.copy()
    df['atr'] = pta.atr(df['high'], df['low'], df['close'], length=50)
