#!/usr/bin/python
# -*- coding: utf-8 -*-

# portfolio.py

###############################################################################
# The Portfolio object handles SignalEvent objects, generate OrderEvent objects
# and interpret FillEvent objects to update positions.
# Can handle position sizing, and current holdings.
###############################################################################

from __future__ import print_function

import datetime
from math import floor
from queue import Queue

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pytrading.event.fill_event import FillEvent
from pytrading.event.order_event import OrderEvent
from pytrading.event.signal_event import SignalEvent
from pytrading.portfolio.performance import create_sharpe_ratio, create_drawdowns
from pytrading.settings.resolution import Daily

plt.style.use("ggplot")


class Portfolio(object):
	"""
	The Portfolio class handles the positions and market value of all
	instruments at a resolution of a "bar", i.e. secondly, minutely, 5-min, 30-
	min, 60-min or EOD.

	The positions DataFrame stores a time-index of the quantity of positions
	held.

	The holdings DataFrame stores the cash and total market holdings value of
	each symbol for a particular time-index, as well as the percentage change in
	portfolio total across bars.
	"""
	def __init__(
			self,
			bars,
			events: Queue,
			start_date,
			resolution,
			initial_capital=10000
	):
		"""
		Initialises the Portfolio object.

		:param bars:
		:param events:
		:param start_date:
		:param resolution:
			The resolution class of the data (e.g. Daily, Hourly, Minutely, etc.)
		:param initial_capital:
		"""
		self.bars = bars  # from DataHandler
		self.events = events
		self.symbol_list = self.bars.symbol_list
		self.start_date = start_date
		self.resolution = resolution
		self.initial_capital = initial_capital

		self.all_positions = self.construct_all_positions()
		self.current_positions = dict(
			(k, v) for k, v in [(s, 0.0) for s in self.symbol_list]
		)
		self.all_holdings = self.construct_all_holdings()
		self.current_holdings = self.construct_current_holdings()

		# Add an equity curve attribute
		self.equity_curve = None

	def construct_all_positions(self):
		"""
		Constructs the positions list using the start_date to determine when the
		time index will begin.

		:return: list of positions
		"""
		d = dict((k, v) for k, v in [(s, 0.0) for s in self.symbol_list])
		d["datetime"] = self.start_date

		return [d]

	def construct_all_holdings(self):
		"""
		Constructs the holdings list using the start_date to determine when the
		time index will begin.

		:return:
		"""
		d = dict((k, v) for k, v in [(s, 0.0) for s in self.symbol_list])
		d["datetime"] = self.start_date
		d["cash"] = self.initial_capital
		d["commission"] = 0.0
		d["total"] = self.initial_capital

		return [d]

	def construct_current_holdings(self):
		"""
		Constructs the dictionary which will hold the instantaneous value of the
		portfolio across all symbols.

		:return:
		"""
		d = dict((k, v) for k, v in [(s, 0.0) for s in self.symbol_list])
		d["cash"] = self.initial_capital
		d["commission"] = 0.0
		d["total"] = self.initial_capital

		return d

	def update_timeindex(self, event):
		"""
		Adds a new record to the positions matrix for the current market data
		bar. This reflects the PREVIOUS bar, i.e. all current market data at
		this stage is known.

		Makes use of a MarketEvent from the events queue.

		:param event:
		:return:
		"""
		# Get the latest datetime
		latest_datetime = self.bars.get_latest_bar_datetime(
			self.symbol_list[0]
		)

		# Update positions
		# ================
		dp = dict((k, v) for k, v in [(s, 0.0) for s in self.symbol_list])
		dp["datetime"] = latest_datetime

		for s in self.symbol_list:
			dp[s] = self.current_positions[s]

		# Append current positions
		self.all_positions.append(dp)

		# Update holdings
		# ===============
		dh = dict((k, v) for k, v in [(s, 0.0) for s in self.symbol_list])
		dh["datetime"] = latest_datetime
		dh["cash"] = self.current_holdings["cash"]
		dh["commission"] = self.current_holdings["commission"]
		dh["total"] = self.current_holdings["cash"]

		# Approximate the real market value
		for s in self.symbol_list:
			market_value = self.current_positions[s] * \
				self.bars.get_latest_bar_value(s, "adj_close")
			dh[s] = market_value
			dh["total"] += market_value

		# Append the current holdings
		self.all_holdings.append(dh)

	def update_positions_from_fill(self, fill: FillEvent):
		"""
		Takes a FillEvent object and updates the position matrix to reflect the
		new position.

		:param fill: The FillEvent object
		:return: None
		"""
		fill_dir = 0
		if fill.direction == "BUY":
			fill_dir = 1
		if fill.direction == "SELL":
			fill_dir = -1

		# Update positions list with new quantities
		self.current_positions[fill.symbol] += (fill_dir * fill.quantity)

	def update_holdings_from_fill(self, fill: FillEvent):
		"""
		Takes a FillEvent object and updates the holdings matrix to reflect the
		holdings value.

		:param fill: The FillEvent object
		:return: None
		"""
		fill_dir = 0
		if fill.direction == "BUY":
			fill_dir = 1
		if fill.direction == "SELL":
			fill_dir = -1

		# Update holdings list with new quantities
		fill_cost = self.bars.get_latest_bar_value(fill.symbol, "adj_close")
		cost = fill_dir * fill_cost * fill.quantity  # This cost is estimated
		self.current_holdings[fill.symbol] += cost
		self.current_holdings["commission"] += fill.commission
		self.current_holdings["cash"] -= (cost + fill.commission)
		self.current_holdings["total"] -= (cost + fill.commission)
		return fill_cost

	def update_fill(self, event: FillEvent):
		"""
		Updates the portfolio current positions and holdings from a FillEvent,
		using update_positions_from_fill() and update_holdings_from_fill().

		:param event: The FillEvent object
		:return: None
		"""
		if event.type == "FILL":
			self.update_positions_from_fill(event)
			self.update_holdings_from_fill(event)

	def generate_order(
			self,
			signal: SignalEvent,
			mkt_quantity=100,
			order_type="MKT"
	):
		"""
		Files an OrderEvent object as a constant quantity sizing of the signal
		object.

		:param signal: The SignalEvent object
		:param mkt_quantity: The quantity of asset, fixed at 100
		:param order_type: The type of order, fixed to be Market Order (MKT)
		:return: OrderEvent object
		"""
		order = None

		symbol = signal.symbol
		direction = signal.signal_type
		strength = signal.strength  # Useful for pairs strategy

		# Check current quantity of the signal's symbol
		cur_quantity = self.current_positions[symbol]

		# Entering the market
		# ===================
		if direction == "LONG" and cur_quantity == 0:
			order = OrderEvent(
				symbol=symbol,
				order_type=order_type,
				quantity=mkt_quantity,
				direction="BUY"
			)

		if direction == "SHORT" and cur_quantity == 0:
			order = OrderEvent(
				symbol=symbol,
				order_type=order_type,
				quantity=mkt_quantity,
				direction="SELL"
			)

		# Exiting the market
		# ==================
		if direction == "EXIT" and cur_quantity > 0:
			order = OrderEvent(
				symbol=symbol,
				order_type=order_type,
				quantity=abs(cur_quantity),
				direction="SELL"
			)

		if direction == "EXIT" and cur_quantity < 0:
			order = OrderEvent(
				symbol=symbol,
				order_type=order_type,
				quantity=abs(cur_quantity),
				direction="BUY"
			)

		return order

	def update_signal(self, event: SignalEvent):
		"""
		Acts on a SignalEvent to generate new orders based on the portfolio
		logic. Adds the generated order to the events queue.

		:param event: The SignalEvent object

		:return: None
		"""
		# If the event type is a SignalEvent, generate an OrderEvent
		if event.type == "SIGNAL":
			order_event = self.generate_order(event)
			self.events.put(order_event)

	def create_equity_curve_dataframe(self):
		"""
		Creates a pandas DataFrame from the all_holdings list of dictionaries.

		:return:
		"""
		curve = pd.DataFrame(self.all_holdings)
		curve.set_index("datetime", inplace=True)
		curve["returns"] = curve["total"].pct_change()
		curve["equity_curve"] = (1.0 + curve["returns"]).cumprod()
		self.equity_curve = curve

	def output_summary_stats(self):
		# Retrieve total return, returns series, and PnL
		total_return = self.equity_curve["equity_curve"][-1]
		returns = self.equity_curve["returns"]
		pnl = self.equity_curve["equity_curve"]

		# Creates a daily period Sharpe ratio
		sharpe_ratio = create_sharpe_ratio(
			returns=returns,
			resolution=self.resolution
		)

		# Create drawdown metrics and add to equity curve dataframe
		drawdown, max_dd, duration_dd = create_drawdowns(pnl=pnl)
		self.equity_curve["drawdown"] = -drawdown

		stats = [
			("Total Return", f"{(total_return - 1.0) * 100.0:.2f}%"),
			("Sharpe Ratio", f"{sharpe_ratio:.2f}"),
			("Maximum Drawdown", f"{-max_dd * 100.0:.2f}%"),
			("Drawdown Duration", f"{duration_dd}")
		]
		self.equity_curve.to_csv("equity.csv")

		return stats

	def output_plot(self):
		fig = plt.figure()
		fig.patch.set_facecolor("white")

		# Equity curve subplot
		ax1 = fig.add_subplot(311, ylabel="Portfolio value\n(normalised)")
		self.equity_curve["equity_curve"].plot(ax=ax1, color="blue", lw=2.)
		ax1.set(xlabel=None)
		plt.grid(True)

		# Returns subplot
		ax2 = fig.add_subplot(312, ylabel="Period returns\n(in decimal pt.)")
		self.equity_curve["returns"].plot(ax=ax2, color="black", lw=2.)
		ax2.set(xlabel=None)
		plt.grid(True)

		# Drawdown subplot
		ax3 = fig.add_subplot(313, ylabel="Drawdowns\n(in decimal pt.)")
		self.equity_curve["drawdown"].plot(ax=ax3, color="red", lw=2.)
		ax3.set(xlabel=None)
		plt.grid(True)

		# Plot the figure
		plt.suptitle("Portfolio Performance Plots", weight="bold")
		plt.tight_layout()
		plt.show()
