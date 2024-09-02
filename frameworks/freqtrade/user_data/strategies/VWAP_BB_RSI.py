# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np
import pandas as pd
from freqtrade.persistence import Trade
from pandas import DataFrame
from datetime import datetime, timedelta
from typing import Optional, Union
from functools import reduce

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IntParameter, IStrategy, merge_informative_pair,
                                stoploss_from_absolute, timeframe_to_prev_date)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib
import re


class VWAP_BB_RSI(IStrategy):
    """
    Strategy motivation: https://www.youtube.com/watch?v=RbQaARxEW9o

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
    # minimal_roi = {
    #     "60": 0.01,
    #     "30": 0.02,
    #     "0": 0.04
    # }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.2
    updated_stoploss = None
    use_custom_stoploss = True

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

    # ==========================================================================
    # BUYING PARAMETERS
    # Bollinger Bands: window and std for buying
    buy_bb_window = IntParameter(7, 21, default=14, space="buy", optimize=True)
    buy_bb_std = DecimalParameter(2.0, 3.0, default=2.0, decimals=1, space="buy", optimize=True)

    # RSI: window
    buy_rsi_window = IntParameter(7, 21, default=16, space="buy", optimize=True)
    buy_rsi_threshold = DecimalParameter(20.0, 50.0, default=45.0, decimals=1, space="buy", optimize=True)

    # VWAP: lookback window, including current candle
    buy_vwap_window = IntParameter(10, 20, default=15, space="buy", optimize=True)

    # ==========================================================================
    # SELLING PARAMETERS
    # ATR
    sell_atr_window = IntParameter(7, 14, default=7, space="sell", optimize=True)
    sell_atr_constant = DecimalParameter(1, 2, default=1.2, decimals=1, space="sell", optimize=True)

    # RSI threshold
    sell_rsi_threshold = IntParameter(80, 90, default=90, space="sell", optimize=True)

    # Reward-risk ratio
    sell_rr_ratio = DecimalParameter(1.0, 5.0, default=1.5, decimals=1, space="sell", optimize=True)

    # ==========================================================================

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = int(
        max(
            buy_bb_window.value,
            buy_vwap_window.value,
            buy_rsi_window.value,
            sell_atr_window.value
        )
    )

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
                'vwap': {},
                'bb_lowerband': {},
                'bb_middleband': {},
                'bb_upperband': {}

            },
            'subplots': {
                # Subplots - each dict defines one additional plot
                "RSI": {
                    'rsi': {},
                },
                "ATR": {
                    'atr': {}
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
        # Set datetime index for the vwap function calculation below
        dataframe.set_index('date', drop=False, inplace=True)

        # RSI
        dataframe['rsi'] = ta.RSI(
            dataframe,
            timeperiod=self.buy_rsi_window.value
        )

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe),
            window=self.buy_bb_window.value,
            stds=self.buy_bb_std.value
        )
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe["bb_percent"] = (
            (dataframe["close"] - dataframe["bb_lowerband"]) /
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
        )
        dataframe["bb_width"] = (
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe["bb_middleband"]
        )

        # VWAP
        dataframe['vwap'] = pta.vwap(
            dataframe['high'],
            dataframe['low'],
            dataframe['close'],
            dataframe['volume']
        )

        # ATR
        dataframe['atr'] = pta.atr(
            dataframe['high'],
            dataframe['low'],
            dataframe['close'],
            length=self.sell_atr_window.value
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        conditions = []

        # Condition 1: all candles need to open/close above the vwap line
        for i in range(self.buy_vwap_window.value-1, -1, -1):
            conditions.append(
                dataframe[['open', 'close']].shift(i).max(axis=1) >=
                dataframe['vwap'].shift(i)
            )

        # Conditions: price closes below the lower Bollinger Band, RSI below
        # threshold, positive volume
        conditions.append(
            (qtpylib.crossed_below(dataframe['close'], dataframe['bb_lowerband'])) &
            (dataframe['rsi'] < self.buy_rsi_threshold.value) &
            (dataframe['volume'] > 0)
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
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
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'exit_long'] = 0

        return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                        current_profit: float, after_fill: bool, **kwargs) -> Optional[float]:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        previous_trade_date = timeframe_to_prev_date(
            self.timeframe,
            trade.open_date_utc - timedelta(minutes=extract_minutes(self.timeframe))
        )

        # Look up the candle before the trade date
        previous_trade_candle = dataframe.loc[
            dataframe['date'] == previous_trade_date]
        if not previous_trade_candle.empty:
            previous_trade_candle = previous_trade_candle.squeeze()

            # Calculate stoploss: (previous close) - alpha * ATR
            stoploss = stoploss_from_absolute(
                stop_rate=previous_trade_candle['close'] - self.sell_atr_constant.value * previous_trade_candle['atr'],
                current_rate=current_rate,
                is_short=trade.is_short
            )

            self.updated_stoploss = stoploss

            return stoploss
        return 100


    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> Optional[Union[str, bool]]:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Exit immediately if RSI is over upper threshold, regardless of profit
        if last_candle['rsi'] >= self.sell_rsi_threshold.value:
            return f'rsi_above_{int(self.sell_rsi_threshold.value)}'
        if self.updated_stoploss is not None:
            if current_profit >= self.sell_rr_ratio.value * abs(self.updated_stoploss):
                return f'{self.sell_rr_ratio.value}:1_rr_ratio'

        previous_trade_date = timeframe_to_prev_date(
            self.timeframe,
            trade.open_date_utc - timedelta(
                minutes=extract_minutes(self.timeframe))
        )

        # Look up the candle before the trade date
        previous_trade_candle = dataframe.loc[
            dataframe['date'] == previous_trade_date]
        if not previous_trade_candle.empty:
            previous_trade_candle = previous_trade_candle.squeeze()
            stop_rate = previous_trade_candle['close'] - self.sell_atr_constant.value * previous_trade_candle['atr']
            previous_trade_candle_close = previous_trade_candle['close']
            stoploss = (previous_trade_candle_close/stop_rate) - 1
            if current_profit >= self.sell_rr_ratio.value * abs(stoploss):
                return f'{self.sell_rr_ratio.value}:1_rr_ratio'


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