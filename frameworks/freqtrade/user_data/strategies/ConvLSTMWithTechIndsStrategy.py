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
                                IntParameter, IStrategy)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib


class ConvLSTMWithTechnicalIndicators(IStrategy):
    """
    Convolutional LSTM mixed Neural Network and technical indicator strategy.
    Inspired by the research of Saul et al. More information can be found on
    https://www.sciencedirect.com/science/article/abs/pii/S0957417420300750

    This strategy will utilise the following technical indicators as features:
      - Accumulation/Distribution Oscillator (A/D)
      - Commodity Channel Index (CCI)
      - Larry William's R (LWI)
      - Momentum
      - Moving average convergence divergence (MACD)
      - Relative Strength Index (RSI)
      - SMA5, SMA10, SMA20, SMA30, SMA60
      - Stochastic D%
      - Stochastic K%
      - Weighted moving average 5, 10, 20, 30, 60


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
    timeframe = '5m'

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "60": 0.01,
        "30": 0.02,
        "0": 0.05
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.05

    # Trailing stoploss
    trailing_stop = True
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # TODO:
    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 60

    # Strategy parameters
    # buy_rsi = IntParameter(10, 40, default=30, space="buy")
    # sell_rsi = IntParameter(60, 90, default=70, space="sell")

    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
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
            'main_plot': {},
            'subplots': {
                # Subplots - each dict defines one additional plot
                "&-target": {
                    "&-target": {"color": "blue"}
                },
                "do_predict": {
                    "do_predict": {"color": "red"}
                }
            }
        }

    def feature_engineering_expand_all(self, dataframe: DataFrame, period,
                                       **kwargs) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        This function will automatically expand the defined features on the config defined
        `indicator_periods_candles`, `include_timeframes`, `include_shifted_candles`, and
        `include_corr_pairs`. In other words, a single feature defined in this function
        will automatically expand to a total of
        `indicator_periods_candles` * `include_timeframes` * `include_shifted_candles` *
        `include_corr_pairs` numbers of features added to the model.

        All features must be prepended with `%` to be recognized by FreqAI internals.

        :param df: strategy dataframe which will receive the features
        :param period: period of the indicator - usage example:
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)
        """
        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        This function will automatically expand the defined features on the config defined
        `include_timeframes`, `include_shifted_candles`, and `include_corr_pairs`.
        In other words, a single feature defined in this function
        will automatically expand to a total of
        `include_timeframes` * `include_shifted_candles` * `include_corr_pairs`
        numbers of features added to the model.

        Features defined here will *not* be automatically duplicated on user defined
        `indicator_periods_candles`

        All features must be prepended with `%` to be recognized by FreqAI internals.

        :param df: strategy dataframe which will receive the features
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-ema-200"] = ta.EMA(dataframe, timeperiod=200)
        """
        n_period = 14
        # Period list for SMA and WMA
        period_list = [5, 10, 20, 30, 60]
        # AD Line
        dataframe["%-ad"] = ta.AD(dataframe)
        # Commodity Channel Index
        dataframe["%-cci"] = ta.CCI(dataframe, timeperiod=n_period)
        # Larry William's %R
        dataframe["%-willr"] = ta.WILLR(dataframe, timeperiod=n_period)
        # Momentum
        dataframe["%-mom"] = ta.MOM(dataframe, timeperiod=n_period)
        # Relative Strength Index
        dataframe["%-rsi"] = ta.RSI(dataframe, timeperiod=n_period)
        # SMA and WMA 5-10-20-30-60
        for n in period_list:
            dataframe[f"%-sma{n}"] = ta.SMA(dataframe, timeperiod=n)
            dataframe[f"%-wma{n}"] = ta.WMA(dataframe, timeperiod=n)
        # Stochastic %K and %D
        # dataframe["%-stoch_k"], dataframe["%-stoch_d"] = ta.STOCH(dataframe)

        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        This optional function will be called once with the dataframe of the base timeframe.
        This is the final function to be called, which means that the dataframe entering this
        function will contain all the features and columns created by all other
        freqai_feature_engineering_* functions.

        This function is a good place to do custom exotic feature extractions (e.g. tsfresh).
        This function is a good place for any feature that should not be auto-expanded upon
        (e.g. day of the week).

        All features must be prepended with `%` to be recognized by FreqAI internals.

        :param df: strategy dataframe which will receive the features
        usage example: dataframe["%-day_of_week"] = (dataframe["date"].dt.dayofweek + 1) / 7
        """
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        Required function to set the targets for the model.
        All targets must be prepended with `&` to be recognized by the FreqAI internals.

        :param df: strategy dataframe which will receive the targets
        usage example: dataframe["&-target"] = dataframe["close"].shift(-1) / dataframe["close"]
        """
        self.freqai.class_names = ["down", "up"]
        dataframe["pct_chg"] = dataframe["close"].shift(-1) / \
                                 dataframe["close"]
        # The target is the change for the next period - up or down
        dataframe["&-target"] = np.where(
            dataframe["pct_chg"] >= 0.0,
            "up",
            "down"
        )
        return dataframe

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

        dataframe = self.freqai.start(dataframe, metadata, self)
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
                (dataframe["&-target"] == "up") &
                (dataframe["do_predict"] == 1) & # Data is trustworthy, i.e. not outlier
                (dataframe["volume"] > 0)  # Make sure Volume is not 0
            ),
            "enter_long"] = 1

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
                    (dataframe["&-target"] == "down") &
                    (dataframe["do_predict"] == 1) &  # Data is trustworthy, i.e. not outlier
                    (dataframe["volume"] > 0)  # Make sure Volume is not 0
            ),
            'exit_long'] = 1

        return dataframe
    