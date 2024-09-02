#!/usr/bin/python
# -*- coding: utf-8 -*-

# performance.py

from __future__ import print_function
import numpy as np
import pandas as pd


def create_sharpe_ratio(returns, resolution, risk_free_rate=0.03):
	"""
	Creates the Sharpe ratio for the strategy, based on a benchmark of zero
	(i.e. no risk-free rate information).

	:param returns:
		A Series or array representing period percentage returns
	:param resolution:
		Daily (252), Hourly (252 * 6.5), Minutely (252 * 6.5 * 60), etc.
	:param risk_free_rate:
		The risk-free rate

	:return: Sharpe ratio
	"""
	periods = resolution().get_n_periods()
	sharpe = np.sqrt(periods) * (np.mean(returns) - np.mean(risk_free_rate)) / np.std(returns)
	return sharpe


def create_drawdowns(pnl: pd.Series):
	"""
	Calculates the largest peak-to-trough drawdown of the PnL curve as well as
	the duration of the drawdown.

	:param pnl:
		pd.Series, period percentage returns
	:return:
		drawdown, duration
	"""
	# Calculates the cumulative returns curve and set up the High Water Mark (hwm)
	hwm = [0]

	# Create the drawdown and duration series
	idx = pnl.index
	drawdown = pd.Series(index=idx)
	duration = pd.Series(index=idx)

	# Loop over the index range
	for t in range(1, len(idx)):
		hwm.append(max(hwm[t-1], pnl[t]))
		drawdown[t] = (hwm[t] - pnl[t])
		duration[t] = (0 if drawdown[t] == 0 else duration[t-1]+1)

	max_drawdown = drawdown.max()
	max_duration = duration.max()
	return drawdown, max_drawdown, max_duration

