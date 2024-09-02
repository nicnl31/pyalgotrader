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

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IntParameter, IStrategy, merge_informative_pair,
                                stoploss_from_absolute, timeframe_to_prev_date)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib
import re


class SmoothedHA_STC_OBV_CHOP_ADX_DI(IStrategy):
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
    timeframe = '1h'

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 0.4
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.20
    use_custom_stoploss = True

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Enable position adjustments
    position_adjustment_enable = True

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # BUY PARAMETERS
    # ==========================================================================
    do_optimize = False

    buy_obv_ema_length = IntParameter(50, 100, default=50, optimize=do_optimize)

    buy_stc_length = IntParameter(10, 12, default=10, optimize=do_optimize)
    buy_stc_fast = IntParameter(23, 26, default=23, optimize=do_optimize)
    buy_stc_slow = IntParameter(45, 50, default=50, optimize=do_optimize)
    buy_stc_threshold_lower = IntParameter(20, 30, default=25, optimize=do_optimize)
    buy_stc_threshold_upper = IntParameter(70, 80, default=75, optimize=do_optimize)
    buy_stc_factor = 0.5

    buy_chop_threshold = DecimalParameter(40.0, 50.0, default=50.0, decimals=1, optimize=do_optimize)

    buy_adx_threshold = DecimalParameter(20.0, 25.0, default=20.0, decimals=1, optimize=do_optimize)

    buy_ha_smooth_length1 = IntParameter(5, 15, default=10, optimize=do_optimize)
    buy_ha_smooth_length2 = IntParameter(5, 15, default=10, optimize=do_optimize)

    # SELL PARAMETERS
    # ==========================================================================
    sell_rr_ratio = DecimalParameter(1.0, 3.0, default=1.5, optimize=do_optimize)

    sell_rr_ratio_divisor = IntParameter(1, 3, default=3, optimize=do_optimize)

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 100

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
                'ha_close_smooth': {
                    'color': '#38393b',
                    'fill_to': 'ha_open_smooth',
                    'fill_color': 'rgba(172, 208, 255, 0.8)'
                },
                'ha_open_smooth': {'color': '#38393b'}
            },
            'subplots': {
                # Subplots - each dict defines one additional plot
                "STC": {
                    'stc': {'color': ''}
                },
                "OBV": {
                    'obv': {'color': ''},
                    f'obv_ema{self.buy_obv_ema_length.value}': {'color': ''}
                },
                "CHOP": {
                    'chop': {'color': '#4e76ed'}
                },
                "ADX": {
                    'adx': {'color': '#f0d84f'},
                    'plus_di': {'color': '#37eb34'},
                    'minus_di': {'color': '#eb3a34'}
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
        # # Set datetime index for the dataframe
        # dataframe.set_index('date', drop=False, inplace=True)

        # Smoothed Heiken Ashi Indicator
        # ------------------------------------
        ha_smooth = heikenashi_smooth(
            dataframe,
            timeperiod1=self.buy_ha_smooth_length1.value,
            timeperiod2=self.buy_ha_smooth_length2.value
        )
        dataframe['ha_open_smooth'] = ha_smooth['ha_open_smooth']
        dataframe['ha_high_smooth'] = ha_smooth['ha_high_smooth']
        dataframe['ha_low_smooth'] = ha_smooth['ha_low_smooth']
        dataframe['ha_close_smooth'] = ha_smooth['ha_close_smooth']

        # ADX-DI_PLUS-DI_MINUS Indicators
        # ------------------------------------
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe)

        # Choppiness Index Indicator
        # ------------------------------------
        dataframe['chop'] = pta.chop(
            dataframe['high'],
            dataframe['low'],
            dataframe['close'],
            length=14
        )

        # Schaff Trend Cycle Indicator
        # ------------------------------------
        stc = pta.stc(dataframe['close'],
                      tclength=self.buy_stc_length.value,
                      fast=self.buy_stc_fast.value,
                      slow=self.buy_stc_slow.value,
                      factor=self.buy_stc_factor)
        dataframe['stc'] = stc[f"STC_{self.buy_stc_length.value}_{self.buy_stc_fast.value}_{self.buy_stc_slow.value}_{self.buy_stc_factor}"]

        # On-balance Volume Indicator
        # ------------------------------------
        dataframe['obv'] = ta.OBV(dataframe)
        dataframe[f'obv_ema{self.buy_obv_ema_length.value}'] = ta.EMA(dataframe['obv'], timeperiod=self.buy_obv_ema_length.value)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        # SCHAFF TREND CYCLE TRIGGER
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['stc'], self.buy_stc_threshold_lower.value)) &
                (dataframe['ha_close_smooth'] > dataframe['ha_open_smooth']) &
                (dataframe['obv'] > dataframe[f'obv_ema{self.buy_obv_ema_length.value}']) &
                (dataframe['chop'] < self.buy_chop_threshold.value) &
                (dataframe['adx'] > self.buy_adx_threshold.value) &
                (dataframe['plus_di'] > dataframe['minus_di']) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'buy_stc_trigger')

        # CHOPPINESS INDEX TRIGGER
        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['chop'],
                                       self.buy_chop_threshold.value)) &
                (dataframe['ha_close_smooth'] > dataframe['ha_open_smooth']) &
                (dataframe['obv'] > dataframe[f'obv_ema{self.buy_obv_ema_length.value}']) &
                (dataframe['stc'] > self.buy_stc_threshold_upper.value) &
                (dataframe['adx'] > self.buy_adx_threshold.value) &
                (dataframe['plus_di'] > dataframe['minus_di']) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'buy_chop_trigger')

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
        """
        Set the custom stoploss as the previous smoothed Heiken Ashi bar's low.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        # Retrieve the time of the candle before the trade candle
        previous_trade_date = timeframe_to_prev_date(
            self.timeframe,
            trade.open_date_utc - timedelta(minutes=extract_minutes(self.timeframe))
        )

        # Retrieve the candle before the trade candle
        previous_trade_candle = dataframe.loc[
            dataframe['date'] == previous_trade_date]
        previous_trade_candle = previous_trade_candle.squeeze()

        stoploss = stoploss_from_absolute(
            stop_rate=previous_trade_candle['ha_low_smooth'],
            current_rate=current_rate,
            is_short=trade.is_short
        )

        return stoploss

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> Optional[Union[str, bool]]:
        """
        Set custom exit based on reward-risk ratio.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        # Retrieve the time of the candle before the trade candle
        previous_trade_date = timeframe_to_prev_date(
            self.timeframe,
            trade.open_date_utc - timedelta(
                minutes=extract_minutes(self.timeframe))
        )

        # Retrieve the candle before the trade candle
        previous_trade_candle = dataframe.loc[
            dataframe['date'] == previous_trade_date]
        previous_trade_candle = previous_trade_candle.squeeze()

        # Calculate the initial stoploss rate
        open_rate = trade.open_rate
        stoploss_rate = abs(previous_trade_candle['ha_low_smooth'] / open_rate - 1)

        if current_profit >= self.sell_rr_ratio.value * stoploss_rate:
            return f'{self.sell_rr_ratio.value}:1_rr'

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:
        """
        Adjust the current position based on the current reward-risk ratio.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)

        # Retrieve the time of the candle before the trade candle
        previous_trade_date = timeframe_to_prev_date(
            self.timeframe,
            trade.open_date_utc - timedelta(
                minutes=extract_minutes(self.timeframe))
        )

        # Retrieve the candle before the trade candle
        previous_trade_candle = dataframe.loc[
            dataframe['date'] == previous_trade_date]
        previous_trade_candle = previous_trade_candle.squeeze()

        # Calculate the initial stoploss and take profit rate
        open_rate = trade.open_rate
        stoploss_rate = abs(
            previous_trade_candle['ha_low_smooth'] / open_rate - 1)
        take_profit_rate = self.sell_rr_ratio.value * stoploss_rate

        partial_exit_stake_amount = -trade.stake_amount / self.sell_rr_ratio_divisor.value
        partial_take_profit_threshold = take_profit_rate / self.sell_rr_ratio_divisor.value

        for i in range(1, self.sell_rr_ratio_divisor.value):
            if (current_profit >= partial_take_profit_threshold * i) and (trade.nr_of_successful_exits == i-1):
                return partial_exit_stake_amount

        return None


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
