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

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib


class KAMA(IStrategy):
    """
    This is a strategy template to get you started.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

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
    minimal_roi = {
        "0": 0.3
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.1

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
    startup_candle_count: int = 121

    # Strategy parameters
    window_length = 21
    num_stdev = 3

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
                'kama': {'color': '#AB924F'}
            },
            'subplots': {
                # Subplots - each dict defines one additional plot
                "KAMA": {
                    'distance': {'color': '#4F58AB'},
                    f'distance_sma{self.window_length}': {'color': '#8DBA56'},
                    f'distance_sma{self.window_length}_upperband': {
                        'color': '#C2E09D'},
                    f'distance_sma{self.window_length}_lowerband': {
                        'color': '#42611D'}
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

        # KAMA100
        dataframe['kama'] = ta.KAMA(dataframe, timeperiod=100)

        # Distance from the close price to the KAMA100
        dataframe['distance'] = dataframe['close'] - dataframe['kama']

        # The SMA of the distance
        dataframe[f'distance_sma{self.window_length}'] = ta.SMA(
            dataframe['distance'],
            timeperiod=self.window_length)

        # The rolling standard deviation of the distance
        dataframe['stdev'] = ta.STDDEV(dataframe['distance'],
                                       timeperiod=self.window_length)

        # Lower and upper bands of the SMA
        dataframe[f'distance_sma{self.window_length}_upperband'] = \
            dataframe[f'distance_sma{self.window_length}'] + \
            self.num_stdev * dataframe['stdev']

        dataframe[f'distance_sma{self.window_length}_lowerband'] = \
            dataframe[f'distance_sma{self.window_length}'] - \
            self.num_stdev * dataframe['stdev']

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
                (qtpylib.crossed_above(
                    dataframe['distance'],
                    dataframe[f'distance_sma{self.window_length}_lowerband'])
                ) &  # Signal: distance crosses above lower band
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'enter_long'] = 1

        dataframe.loc[
            (
                    (qtpylib.crossed_below(
                        dataframe['distance'],
                        dataframe[
                            f'distance_sma{self.window_length}_upperband'])
                    ) &  # Signal: distance crosses below upper band
                    (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'enter_short'] = 1

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
                (qtpylib.crossed_below(
                    dataframe['close'],
                    dataframe['kama'])) &  # Signal: distance crosses below upper band
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'exit_long'] = 1

        # Uncomment to use shorts (Only used in futures/margin mode. Check the documentation for more info)
        dataframe.loc[
            (
                (qtpylib.crossed_above(
                    dataframe['close'],
                    dataframe['kama'])) &  # Signal: distance crosses below upper band
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'exit_short'] = 1

        return dataframe
    