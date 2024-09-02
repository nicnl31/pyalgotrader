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
                                IntParameter, IStrategy, informative,
                                merge_informative_pair)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib


class SMAInvestV1(IStrategy):
    """
    This SMA Investing strategy is motivated by
    https://medium.datadriveninvestor.com/i-used-this-one-indicator-to-buy-
    stocks-that-moved-up-80-when-the-so-called-experts-called-a-848fa159914c

    It uses the 4h and 1d SMA200 of the Bitcoin price to

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
    informative_timeframe = '1d'

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    # minimal_roi = {
    #     "60": 0.01,
    #     "30": 0.02,
    #     "0": 0.04
    # }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.3

    # Trailing stoploss
    trailing_stop = True
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 1200

    # Optional order type mapping.
    order_types = {
        'entry': 'market',
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
            'main_plot': {},
            'subplots': {
                f"BTC/USDT {self.timeframe}": {
                    f'close_{self.timeframe}': {'color': '#3137E8'},
                    f'sma200_{self.timeframe}': {'color': '#3234A8'}
                },
                f"BTC/USDT {self.informative_timeframe}": {
                    f'close_{self.informative_timeframe}': {'color': '#30AB4F'},
                    f'sma200_{self.informative_timeframe}': {'color': '#A86032'}
                }
            }
        }

    def informative_pairs(self):
        return [
            ("BTC/USDT", self.timeframe),
            ("BTC/USDT", self.informative_timeframe)
        ]

    # @informative(timeframe, 'BTC/USDT')
    # @informative(informative_timeframe, 'BTC/USDT')
    # def populate_indicators_btc_inf(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    #     """
    #     Adds several different TA indicators to the given DataFrame
    #
    #     Performance Note: For the best performance be frugal on the number of indicators
    #     you are using. Let uncomment only the indicator you are using in your strategies
    #     or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
    #     :param dataframe: Dataframe with data from the exchange
    #     :param metadata: Additional information, like the currently traded pair
    #     :return: a Dataframe with all mandatory indicators for the strategies
    #     """
    #
    #     # SMA - Simple Moving Average
    #     dataframe['sma200'] = ta.SMA(dataframe, timeperiod=200)
    #
    #     return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        # Don't do anything if DataProvider is not available.
        if not self.dp:
            return dataframe

        # Get the 4h informative BTC/USDT data
        informative_4h = self.dp.get_pair_dataframe(
            pair="BTC/USDT",
            timeframe=self.timeframe
        )
        informative_4h["sma200"] = ta.SMA(informative_4h, timeperiod=200)

        # Get the 1d informative BTC/USDT data
        informative_1d = self.dp.get_pair_dataframe(
            pair="BTC/USDT",
            timeframe=self.informative_timeframe
        )
        informative_1d["sma200"] = ta.SMA(informative_1d, timeperiod=200)

        # Merge informative pairs
        dataframe = merge_informative_pair(dataframe, informative_4h,
                                           self.timeframe, self.timeframe,
                                           ffill=True)
        dataframe = merge_informative_pair(dataframe, informative_1d,
                                           self.timeframe,
                                           self.informative_timeframe,
                                           ffill=True)

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
                # 1: BTC/USDT price above main timeframe
                (
                    dataframe[f'close_{self.timeframe}'] >
                    dataframe[f'sma200_{self.timeframe}']) &

                # 2: BTC/USDT price above informative timeframe
                (
                    dataframe[f'close_{self.informative_timeframe}'] >
                    dataframe[f'sma200_{self.informative_timeframe}']) &

                # 3: Volume is not 0
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
                # 1: BTC/USDT price crossing below main timeframe
                (
                    dataframe[f'close_{self.timeframe}'] <
                    dataframe[f'sma200_{self.timeframe}']) &

                # 2: Volume is not 0
                (dataframe['volume'] > 0)
            ),
            'exit_long'] = 1

        return dataframe
    