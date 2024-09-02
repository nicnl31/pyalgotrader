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
from functools import reduce

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IntParameter, IStrategy, merge_informative_pair)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from technical import qtpylib


class BBRSI_Informative(IStrategy):
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
    inf_timeframe = '4h'

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 0.15
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.10

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

    # Strategy parameters
    buy_ema = IntParameter(150, 200, default=200, space="buy", optimize=False)
    buy_above_ema_window = IntParameter(5, 15, default=6, space="buy", optimize=True)
    buy_bb_window = IntParameter(7, 21, default=20, space="buy", optimize=False)
    buy_bb_std = DecimalParameter(2, 3, default=2.5, decimals=1, optimize=False)
    sell_rsi = IntParameter(65, 90, default=76, space="sell", optimize=True)
    sell_unclog = IntParameter(5, 10, default=10, space="sell", optimize=True)

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = int(
        max(
            buy_ema.value,
            buy_above_ema_window.value,
            buy_bb_window.value
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
                'bb_lowerband': {},
                'bb_upperband': {},
                f'ema{self.buy_ema.value}': {}
            },
            'subplots': {
                # Subplots - each dict defines one additional plot
                "RSI": {
                    'rsi': {}
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
        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()
        # Assign tf to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, self.inf_timeframe) for pair in pairs]
        return informative_pairs

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
        if not self.dp:
            # Don't do anything if DataProvider is not available.
            return dataframe

        # Get the informative pairs
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'],
                                                 timeframe=self.inf_timeframe)
        # Get the EMA200 for the informative pairs
        informative[f'ema{self.buy_ema.value}'] = ta.EMA(informative, timeperiod=self.buy_ema.value)

        # Use the helper function merge_informative_pair to safely merge the
        # pairs. Automatically renames the columns and merges a shorter
        # timeframe dataframe and a longer timeframe informative pair.
        # Use 'ffill' to have the informative pair value available in every row
        # throughout the day. Without this, comparisons between columns of the
        # original and the informative pair would only work once per day.
        dataframe = merge_informative_pair(
            dataframe,
            informative,
            self.timeframe,
            self.inf_timeframe,
            ffill=True
        )

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe),
            window=self.buy_bb_window.value,
            stds=self.buy_bb_std.value)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_upperband'] = bollinger['upper']

        # # EMA - Exponential Moving Average
        dataframe[f'ema{self.buy_ema.value}'] = ta.EMA(dataframe, timeperiod=self.buy_ema.value)

        # print(dataframe.head(20))

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        conditions_1, conditions_2 = [], []

        # Condition 1: All candles within the specified window need to open and
        # close above the EMA line
        for i in range(self.buy_above_ema_window.value-1, -1, -1):
            conditions_1.append(
                dataframe[['open', 'close']].shift(i).min(axis=1) >=
                dataframe[f'ema{self.buy_ema.value}'].shift(i)
            )

        num_hrs_inf_tf = int(self.inf_timeframe[:-1])
        for i in range(
                self.buy_above_ema_window.value * num_hrs_inf_tf - num_hrs_inf_tf,
                -1,
                -num_hrs_inf_tf
        ):
            conditions_1.append(
                dataframe[[f'open_{self.inf_timeframe}', f'close_{self.inf_timeframe}']].shift(i).min(axis=1) >=
                dataframe[f'ema{self.buy_ema.value}_{self.inf_timeframe}'].shift(i)
            )

        # Condition 2: The trigger candle must cross below the lower BB
        conditions_1.append(
            qtpylib.crossed_below(dataframe['close'], dataframe['bb_lowerband'])
        )

        # conditions_2.append(qtpylib.crossed_below(dataframe['rsi'], 10))

        # Condition 3: Volume is greater than 0
        conditions_1.append((dataframe['volume'] > 0))
        # conditions_2.append((dataframe['volume'] > 0))

        if conditions_1:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions_1),
                ['enter_long', 'enter_tag']] = (1, 'enter_ema_bb')

        # if conditions_2:
        #     dataframe.loc[
        #         reduce(lambda x, y: x & y, conditions_2),
        #         ['enter_long', 'enter_tag']] = (1, 'enter_rsi_only')

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
                # Condition 1: RSI crosses above sell_rsi value
                (qtpylib.crossed_above(dataframe['rsi'], self.sell_rsi.value)) &
                # Condition 2: Volume is greater than 0
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'exit_long'] = 1

        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> Optional[Union[str, bool]]:
        # Sell any positions if they are held for more than a specific number of
        # days
        if (
                ((current_time - trade.open_date_utc).days >= self.sell_unclog.value) &
                (trade.enter_tag == 'enter_ema_bb')
        ):
            return 'unclog_ema_bb'
        if (
                ((current_time - trade.open_date_utc).days >= self.sell_unclog.value + 4) &
                (trade.enter_tag == 'enter_rsi_only')
        ):
            return 'unclog_rsi_only'
