# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IntParameter, IStrategy, merge_informative_pair)
from freqtrade.exchange import timeframe_to_prev_date


# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib


class AlligatorV3(IStrategy):
    """
    Inspired by: https://www.youtube.com/watch?v=BY9kQPy_XQc

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_entry_trend, populate_exit_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = '4h'

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    # minimal_roi = {
    #     "1440": 0.0,
    #     "720": 0.05,
    #     "0": 0.10
    # }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.15

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 220

    # Strategy parameters
    # buy_rsi = IntParameter(10, 40, default=30, space="buy")
    # sell_rsi = IntParameter(60, 90, default=70, space="sell")

    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }
    
    @property
    def plot_config(self):
        return {
            # Main plot indicators (Moving averages, ...)
            'main_plot': {
                'lips': {'color': '#3ADE4D'},
                'teeth': {'color': '#3A4DDE'},
                'jaw': {'color': '#C91616'},
                'trendline': {'color': '#EDE84A'}
            },
            'subplots': {
                "RSI": {
                    'rsi': {'color': 'red'}
                }
            }
        }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """

        # Lips: SMA5
        dataframe['lips'] = ta.SMA(dataframe, timeperiod=5)

        # Teeth: SMA8
        dataframe['teeth'] = ta.SMA(dataframe, timeperiod=8)

        # Jaw: SMA13
        dataframe['jaw'] = ta.SMA(dataframe, timeperiod=13)

        # Trend line: EMA200
        dataframe['trendline'] = ta.EMA(dataframe, timeperiod=200)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        dataframe.loc[
            (
                # SMA5 > SMA8 > SMA13 and all are above the long term trend
                (qtpylib.crossed_above(dataframe['lips'], dataframe['teeth'])) &
                (dataframe['teeth'] > dataframe['jaw']) &
                (dataframe['close'] > dataframe['trendline']) &

                # # Pullback
                # (dataframe['close'] < dataframe['close'].shift(3)) &

                # # Breakout above the lips
                # (qtpylib.crossed_above(dataframe['close'], dataframe['lips'])) &

                # Confirmed trend: RSI > 50
                (dataframe['rsi'] > 50) &

                # Volume
                (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """
        dataframe.loc[
            (
                (
                    (
                        (
                            (qtpylib.crossed_below(dataframe['close'], dataframe['jaw'])) |
                            (qtpylib.crossed_below(dataframe['close'], dataframe['trendline']))
                        ) &
                        (dataframe['jaw'] < dataframe['teeth'])
                    ) |
                    (qtpylib.crossed_below(dataframe['rsi'], 53))
                ) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'exit_long'] = 1

        return dataframe
    