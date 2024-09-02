#!/usr/bin/python
# -*- coding: utf-8 -*-

# backtest_session.py

from __future__ import print_function
import datetime
import pprint
import matplotlib.pyplot as plt

from queue import Queue, Empty
import time

from pytrading.trading.session import TradingSession

plt.style.use("ggplot")


class BacktestTradingSession(TradingSession):
	"""
	Encapsulates the settings and components for carrying out an event-driven
	backtest trading session.
	"""
	def __init__(
			self,
			symbol_list: list,
			initial_capital: float,
			heartbeat: float,
			start_date: datetime.datetime,
			data_resolution,
			data_handler,
			execution_handler,
			portfolio,
			strategy
	):
		"""
		Initialises the backtest session.

		:param symbol_list:
			The list of symbol strings
		:param initial_capital:
			The starting capital for the portfolio
		:param heartbeat:
			Backtest "heartbeat" in seconds
		:param start_date:
			The start datetime of the strategy
		:param data_resolution:
			The resolution of the data, i.e. the time interval (Daily, Hourly,
			Minutely, etc.)
		:param data_handler:
			Handles the market data feed
		:param execution_handler:
			Handles the orders/fills for trades, and brokerage communications
		:param portfolio:
			Keeps track of portfolio current and prior positions
		:param strategy:
			Generates signals based on market data
		"""
		self.symbol_list = symbol_list
		self.initial_capital = initial_capital
		self.heartbeat = heartbeat
		self.start_date = start_date
		self.data_resolution = data_resolution

		self.data_handler_cls = data_handler
		self.execution_handler_cls = execution_handler
		self.portfolio_cls = portfolio
		self.strategy_cls = strategy

		# Creates empty queue object
		self.events = Queue()

		self.signals = 0
		self.orders = 0
		self.fills = 0
		self.num_strategies = 1

		self._generate_trading_instances()

	def _generate_trading_instances(self):
		"""
		Attaches the trading objects (DataHandler, Strategy, Portfolio,
		ExecutionHandler) to various internal members.

		:return: None
		"""
		print("Creating DataHandler...")
		self.data_handler = self.data_handler_cls(
			self.events,
			self.symbol_list,
			self.start_date,
			self.data_resolution
		)
		print("Creating Strategy...")
		self.strategy = self.strategy_cls(self.data_handler, self.events)
		print("Creating Portfolio...")
		self.portfolio = self.portfolio_cls(
			self.data_handler,
			self.events,
			self.start_date,
			self.data_resolution,
			self.initial_capital
		)
		print("Creating ExecutionHandler...")
		self.execution_handler = self.execution_handler_cls(self.events)

	def _run_backtest(self):
		"""
		Executes the backtest.

		:return:
		"""
		i = 0
		while True:
			i += 1
			if self.data_handler.continue_backtest:
				self.data_handler.update_bars()
			else:
				break

			# Handle the events
			while True:
				try:
					event = self.events.get(block=False)
				except Empty:
					break
				else:
					if event is not None:
						if event.type == "MARKET":
							self.strategy.calculate_signals(event)
							self.portfolio.update_timeindex(event)
						elif event.type == "SIGNAL":
							self.signals += 1
							self.portfolio.update_signal(event)
						elif event.type == "ORDER":
							self.orders += 1
							self.execution_handler.execute_order(event)
							# event.print_order()
						elif event.type == "FILL":
							self.fills += 1
							self.portfolio.update_fill(event)
			time.sleep(self.heartbeat)

	def _output_performance(self):
		"""
		Outputs the strategy performance from the backtest.

		:return: None
		"""
		self.portfolio.create_equity_curve_dataframe()

		print("Creating summary statistics...")
		stats = self.portfolio.output_summary_stats()

		print("Creating equity curve...")
		print(self.portfolio.equity_curve.tail(10))
		pprint.pprint(stats)

		print(f"Signals: {self.signals}")
		print(f"Orders: {self.orders}")
		print(f"Fills: {self.fills}")

		self.portfolio.output_plot()

	def output_trade_signals(self):
		"""
		Outputs the trade signals for each symbol in the backtest. Use adjusted
		close price for plots.

		:return: Plots, one for each symbol.
		"""
		for s in self.symbol_list:
			plt.plot(self.data_handler.symbol_data_full[s]["adj_close"])
			for signal, data in self.strategy.trades[s].groupby("signal_type"):
				plt.scatter(
					x=data.index,
					y=data["price"],
					label=signal,
					marker=lambda m: "^" if signal == "LONG" else "v"
				)
			plt.title(f"Stock: {s}")
			plt.show()

	def run(self):
		"""
		Runs the backtest and outputs portfolio performance.

		:return: None
		"""
		self._run_backtest()
		self._output_performance()
