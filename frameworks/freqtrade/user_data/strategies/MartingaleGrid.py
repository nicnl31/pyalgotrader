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
from freqtrade.persistence import Trade

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib


class MartingaleGrid(IStrategy):
    """
    This strategy is motivated by the Martingale risk management strategy by
    Ryan Brown. More info here: https://www.youtube.com/watch?v=Wbi9Knt5Yrk

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
    timeframe = '30m'

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    # minimal_roi = {
    #     "0": 0.1,
    #     "1440": 0.09,
    #     "2880": 0.08,
    #     "4320": 0.07,
    #     "5760": 0.06,
    #     "7200": 0.05
    # }
    minimal_roi = {
        "0": 0.198,
        "86": 0.177,
        "413": 0.068,
        "750": 0
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.299

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 0

    # Enable position adjustments
    position_adjustment_enable = True

    # Strategy parameters
    initial_stake_ratio = DecimalParameter(0.001, 0.1, decimals=3, default=0.016, space='buy', optimize=True)  # 0.1% of the total balance
    martingale_multiplier = DecimalParameter(1.25, 2, decimals=2, default=1.52, space='buy', optimize=True)
    num_gridlines = IntParameter(3, 10, default=4, space='buy', optimize=True)
    grid_bottom = 0.9 * stoploss

    grid = np.linspace(0.0, grid_bottom, num_gridlines.value+1)
    delta = grid[0] - grid[1]
    grid = np.delete(grid, 0)

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
            'main_plot': {},
            'subplots': {}
        }

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
                # (dataframe['close'] > dataframe['trendline']) &
                # (qtpylib.crossed_above(
                #     dataframe['rsi'],
                #     30
                # )) &
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
                (dataframe['volume'] > 0)
            ),
            'exit_long'] = 0

        return dataframe

    # This is called when placing the initial order (opening trade)
    def custom_stake_amount(self, pair: str, current_time: datetime,
                            current_rate: float,
                            proposed_stake: float,
                            min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str],
                            side: str,
                            **kwargs) -> float:

        return max(5.0, self.wallets.get_total_stake_amount() * self.initial_stake_ratio.value)

    # def adjust_trade_position(self, trade: 'Trade', current_time: datetime,
    #                           current_rate: float, current_profit: float,
    #                           min_stake: Optional[float], max_stake: float,
    #                           current_entry_rate: float, current_exit_rate: float,
    #                           current_entry_profit: float, current_exit_profit: float,
    #                           **kwargs) -> Optional[float]:
    #     """
    #     Custom trade adjustment logic, returning the stake amount that a trade should be
    #     increased or decreased.
    #     This means extra entry or exit orders with additional fees.
    #     Only called when `position_adjustment_enable` is set to True.
    #
    #     For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/
    #
    #     When not implemented by a strategy, returns None
    #
    #     :param trade: trade object.
    #     :param current_time: datetime object, containing the current datetime
    #     :param current_rate: Current entry rate (same as current_entry_profit)
    #     :param current_profit: Current profit (as ratio), calculated based on current_rate
    #                             (same as current_entry_profit).
    #     :param min_stake: Minimal stake size allowed by exchange (for both entries and exits)
    #     :param max_stake: Maximum stake allowed (either through balance, or by exchange limits).
    #     :param current_entry_rate: Current rate using entry pricing.
    #     :param current_exit_rate: Current rate using exit pricing.
    #     :param current_entry_profit: Current profit using entry pricing.
    #     :param current_exit_profit: Current profit using exit pricing.
    #     :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
    #     :return float: Stake amount to adjust your trade,
    #                    Positive values to increase position, Negative values to decrease position.
    #                    Return None for no action.
    #     """
    #     print(current_time, f'current_profit:{round(current_profit * 100, 2)}')
    #     if current_profit > self.grid[0]:
    #         return None
    #     # Retrieve the list of filled entries
    #     filled_entries = trade.select_filled_orders(trade.entry_side)
    #     count_of_entries = trade.nr_of_successful_entries
    #
    #     print('filled_entries', filled_entries)
    #     print('count_of_entries', count_of_entries)
    #     print('trade', trade)
    #
    #     try:
    #         # Retrieve the list of stake amounts for filled entries
    #         stake_list = [
    #             filled_entries[i].stake_amount
    #             for i in range(len(filled_entries))
    #         ]
    #
    #         # Get the stake amount of the first entry
    #         initial_stake_amount = stake_list[0]
    #
    #         # Implement Martingale logic: if losing and there's not already an
    #         # order at the specified stake amount, then buy more
    #
    #         if current_profit <= self.grid[0]:
    #             for i in range(1, len(self.grid)+1):
    #                 if (
    #                         (self.grid[i-1] < current_profit < self.grid[i]) and
    #                         (initial_stake_amount * self.martingale_multiplier**(i+1) not in stake_list)
    #                 ):
    #                     return initial_stake_amount * self.martingale_multiplier**(i+1)
    #
    #     except Exception as exception:
    #         return None
    #
    #     return None

    def adjust_trade_position(self, trade: 'Trade', current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: Optional[float],
                              max_stake: float, **kwargs):
        # print(current_time, f'current_profit:{round(current_profit*100, 2)}')

        if current_profit > -self.delta:
            return None

        filled_entries = trade.select_filled_orders(trade.entry_side)

        # print('filled_entries', filled_entries)
        # print('count_of_entries', count_of_entries)
        # print('trade', trade)

        # dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        #
        # # In dry/live runs trade open date will not match candle open date therefore it must be
        # # rounded.
        # trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        # # Look up trade candle.
        # trade_candle = dataframe.loc[dataframe['date'] == trade_date]
        # # trade_candle may be empty for trades that just opened as it is still incomplete.
        # if not trade_candle.empty:
        #     trade_candle = trade_candle.squeeze()

        try:
            stake_amount = filled_entries[0].stake_amount
            profit_from_open = (current_rate - trade.open_rate) / trade.open_rate

            # print(f'trade open rate: {trade.open_rate}')
            # print(f'current rate: {current_rate}')
            # print(f'profit from open: {profit_from_open}')

            for i in range(1, len(self.grid)+1):
                if self.grid[i - 1] > profit_from_open > self.grid[i]:
                    stake_amount *= self.martingale_multiplier.value**(i+1)
                    # print('new stake_amount:', stake_amount)
            return stake_amount
        except Exception as exception:
            return None

        return None
