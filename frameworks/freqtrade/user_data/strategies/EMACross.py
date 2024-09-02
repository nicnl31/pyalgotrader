# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np
import pandas as pd
from freqtrade.persistence import Trade
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IntParameter, IStrategy, merge_informative_pair,
                                stoploss_from_absolute, timeframe_to_prev_date)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib


class EMACross(IStrategy):
    """
    This strategy uses the three EMAs together with MACD and the ADX to
    determine buying signals.

    ============================================================================
    THE SETUP
    1. EMAs:
       - Fast period: 5
       - Medium period: 20
       - Slow period: 50
    2. MACD:
       - Fast period: 12
       - Slow period: 26
       - Signal period: 9
    3. ADX:
       - Periods: 14

    ============================================================================
    BUYING
    The strategy buys when:
    (EMA: Fast > Medium > Slow)
       AND
    (
       (
          (MACD histogram crosses above 0) AND (ADX > 20)
       )
       OR
       (
          (ADX crosses above 20) AND (MACD > 0)
       )
    )



    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = '30m'

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 0.2
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.05
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

    # Strategy parameters
    buy_ema_short = IntParameter(3, 11, default=5, space="buy")
    buy_ema_medium = IntParameter(11, 26, default=20, space="buy")
    buy_ema_long = IntParameter(34, 55, default=50, space="buy")
    buy_adx_crossover = DecimalParameter(20.0, 30.0, decimals=1, default=20.0, space="buy")
    profit_risk_ratio = 3

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = int(np.ceil(2 * buy_ema_long.value))

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
                f'ema{self.buy_ema_short.value}': {'color': '#3458EB'},
                f'ema{self.buy_ema_medium.value}': {'color': '#EB3455'},
                f'ema{self.buy_ema_long.value}': {'color': '#EBDE34'}
            },
            'subplots': {
                "MACD": {
                    'macd': {'color': 'blue', 'fill_to': 'macdhist'},
                    'macdsignal': {'color': 'orange'},
                    'macdhist': {'type': 'bar', 'plotly': {'opacity': 0.9}}
                },
                "ADX": {
                    'adx': {'color': '#EB4634'}
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
        # # EMA - Exponential Moving Average
        dataframe[f'ema{self.buy_ema_short.value}'] = ta.EMA(
            dataframe,
            timeperiod=self.buy_ema_short.value)
        dataframe[f'ema{self.buy_ema_medium.value}'] = ta.EMA(
            dataframe,
            timeperiod=self.buy_ema_medium.value)
        dataframe[f'ema{self.buy_ema_long.value}'] = ta.EMA(
            dataframe,
            timeperiod=self.buy_ema_long.value)

        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # ADX
        dataframe['adx'] = ta.ADX(dataframe)

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
                (
                    dataframe[f'ema{self.buy_ema_short.value}'] >
                    dataframe[f'ema{self.buy_ema_medium.value}']
                ) &
                (
                    dataframe[f'ema{self.buy_ema_medium.value}'] >
                    dataframe[f'ema{self.buy_ema_long.value}']
                ) &
                (
                    (
                        (qtpylib.crossed_above(dataframe['macdhist'],0.0)) &
                        (dataframe['adx'] > self.buy_adx_crossover.value) &
                        (dataframe['adx'] > dataframe['adx'].shift(3))  # ADX has to be increasing
                    ) |
                    (
                        (qtpylib.crossed_above(dataframe['adx'], self.buy_adx_crossover.value)) &
                        (dataframe['macd'] > 0.0) &
                        (dataframe['macdhist'] > 0.0)
                    )
                ) &

                (dataframe['volume'] > 0)  # Make sure Volume is not 0
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
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'exit_long'] = 0

        return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                        current_profit: float, after_fill: bool, **kwargs) -> Optional[float]:
        """
        Custom stoploss logic, returning the new distance relative to current_rate (as ratio).
        e.g. returning -0.05 would create a stoploss 5% below current_rate.
        The custom stoploss can never be below self.stoploss, which serves as a hard maximum loss.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns the initial stoploss value
        Only called when use_custom_stoploss is set to True.

        :param pair: Pair that's currently analyzed
        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in ask_strategy.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: New stoploss value, relative to the currentrate
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        # Look up trade candle.
        trade_candle = dataframe.loc[dataframe['date'] == trade_date]
        # trade_candle may be empty for trades that just opened as it is still incomplete.
        if not trade_candle.empty:
            trade_candle = trade_candle.squeeze()
            stoploss = stoploss_from_absolute(
                stop_rate=trade_candle[f'ema{self.buy_ema_long.value}'],
                current_rate=current_rate,
                is_short=trade.is_short
            )
            return stoploss
        # return some value that won't cause stoploss to update
        return 100

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> Optional[Union[str, bool]]:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        trade_candle = dataframe.loc[dataframe['date'] == trade_date]
        if not trade_candle.empty:
            trade_candle = trade_candle.squeeze()
            open_ema_long = trade_candle[f'ema{self.buy_ema_long.value}']
            open_rate = trade.open_rate
            stoploss_rate = open_rate/open_ema_long - 1
            if current_profit > self.profit_risk_ratio * stoploss_rate:
                return f'{self.profit_risk_ratio}:1_roi'

    