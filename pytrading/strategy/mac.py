#!/usr/bin/python
# -*- coding: utf-8 -*-

# mac.py

###############################################################################
# Implementation of the Moving Average Crossover strategy
###############################################################################

from __future__ import print_function
import datetime
from datetime import timezone
import numpy as np
import pandas as pd
from queue import Queue


from pytrading.strategy.strategy import Strategy
from pytrading.event.signal_event import SignalEvent
from pytrading.trading.backtest_session import BacktestTradingSession
from pytrading.data.data import DataHandler
from pytrading.data.yahoo_data import YahooFinanceDataHandler
from pytrading.execution.simulated_execution import SimulatedExecutionHandler
from pytrading.portfolio.portfolio import Portfolio
from pytrading.settings.resolution import Daily, Hourly


class MovingAverageCrossStrategy(Strategy):
	"""
	Implementation of the Moving Average Crossover strategy, with a short/long
	simple weighted moving average. Default short/long windows are 10/50
	periods, respectively.
	"""
	def __init__(
			self,
			bars: DataHandler,
			events: Queue,
			short_window=10,
			long_window=50
	):
		"""
		Initialises the Moving Average Crossover strategy.

		:param bars: DataHandler
			The DataHandler object that provides bar information
		:param events: Queue
			The event Queue object
		:param short_window: int
			The short moving average lookback
		:param long_window: int
			The long moving average lookback
		"""
		self.bars = bars
		self.symbol_list = self.bars.symbol_list
		self.events = events
		self.short_window = short_window
		self.long_window = long_window

		# Initiate an empty trade dataset, which records the trades done
		self.trades = {}
		for s in self.symbol_list:
			self.trades[s] = pd.DataFrame(
				columns=["signal_type", "price"],
				index=pd.DatetimeIndex(data=[], name="datetime")
			)

		# "bought" tells the strategy when the backtest is "in the market". Its
		# status are one of "OUT", "LONG", "SHORT".
		self.bought = self._calculate_initial_bought()

	def _calculate_initial_bought(self):
		"""
		Adds keys to the "bought" dictionary for all symbols and sets them to
		"OUT".

		Explanation: The strategy begins out of the market, so we set all
		initial bought values to be "OUT".

		:return: dict
			Bought
		"""
		bought = {}
		for s in self.symbol_list:
			bought[s] = "OUT"

		return bought

	def calculate_signals(self, event):
		"""
		Generates a new set of signals based on the simple moving average
		crossover.

		For "LONG" entry: short window crossing above long window
		For "OUT" entry: vice versa

		:param event: MarketEvent
			A MarketEvent object
		:return:
		"""
		if event.type == "MARKET":
			for s in self.symbol_list:
				# Get the newest bar values for the long window, which also
				# covers the short window
				bars = self.bars.get_latest_bars_values(
					symbol=s,
					val_type="adj_close",
					N=self.long_window
				)
				bar_date = self.bars.get_latest_bar_datetime(symbol=s)
				bar_price = self.bars.get_latest_bar_value(
					symbol=s,
					val_type="adj_close"
				)
				if bars is not None and len(bars) > 0:
					short_sma = np.mean(bars[-self.short_window:])
					long_sma = np.mean(bars[-self.long_window:])

					symbol = s
					dt = datetime.datetime.now(timezone.utc)
					sig_dir = ""

					if short_sma > long_sma and self.bought[s] == "OUT":
						sig_dir = "LONG"
						signal = SignalEvent(
							strategy_id=1,
							symbol=symbol,
							datetime=dt,
							signal_type=sig_dir,
							strength=1.0
						)
						print(f"{bar_date}: {sig_dir} {symbol} at {bar_price:.2f}")
						self.events.put(signal)
						self.bought[s] = "LONG"
					elif short_sma < long_sma and self.bought[s] == "LONG":
						sig_dir = "EXIT"
						signal = SignalEvent(
							strategy_id=1,
							symbol=symbol,
							datetime=dt,
							signal_type=sig_dir,
							strength=1.0
						)
						print(f"{bar_date}: {sig_dir} {symbol} at {bar_price:.2f}")
						self.events.put(signal)
						self.bought[s] = "OUT"
					# Insert data into trades dataset for the symbol
					self.trades[symbol].loc[bar_date] = (sig_dir, bar_price)


def main(
		symbol_list,
		start_date,
		data_resolution,
		initial_capital=10000.0,
		heartbeat=0.0
):
	backtest = BacktestTradingSession(
		symbol_list=symbol_list,
		start_date=start_date,
		initial_capital=initial_capital,
		heartbeat=heartbeat,
		data_resolution=data_resolution,
		data_handler=YahooFinanceDataHandler,
		execution_handler=SimulatedExecutionHandler,
		portfolio=Portfolio,
		strategy=MovingAverageCrossStrategy
	)
	backtest.run()
	backtest.output_trade_signals()


if __name__ == "__main__":
	main(
		symbol_list=["ETH-USD"],
		start_date=datetime.datetime(2023, 1, 1, 0, 0, 0),
		data_resolution=Daily
	)
