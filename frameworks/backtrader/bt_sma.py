from datetime import datetime
import backtrader as bt
from data.get_data_yahoo import get_data

# Create a subclass of SignaStrategy to define the indicators and signals

class SmaCross(bt.SignalStrategy):
	# list of parameters which are configurable for the strategy
	params = dict(
		pfast=10,  # period for the fast moving average
		pslow=30  # period for the slow moving average
	)

	def __init__(self):
		sma1 = bt.ind.SMA(period=self.p.pfast)  # fast moving average
		sma2 = bt.ind.SMA(period=self.p.pslow)  # slow moving average
		crossover = bt.ind.CrossOver(sma1, sma2)  # crossover signal
		self.signal_add(bt.SIGNAL_LONG, crossover)  # use it as LONG signal


if __name__ == "__main__":
	cerebro = bt.Cerebro()  # create a "Cerebro" engine instance

	# Create a data feed
	start = datetime(2013, 1, 1)
	end = datetime.now()
	ticker = "CBA.AX"
	data = bt.feeds.PandasData(
		dataname=get_data(ticker, start, end)
	)

	# First component of Cerebro: the data
	cerebro.adddata(data)  # Add the data feed

	# Second component of Cerebro: the strategy
	cerebro.addstrategy(SmaCross)  # Add the trading strategy
	cerebro.run()  # run it all
	cerebro.plot()  # and plot it with a single command
