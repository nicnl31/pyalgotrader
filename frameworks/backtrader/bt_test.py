from __future__ import (
	absolute_import,
	division,
	print_function,
	unicode_literals
)
from datetime import datetime
import backtrader as bt
from data.get_data_yahoo import get_data


class TestStrategy(bt.Strategy):
	def log(self, txt, dt=None):
		"""
		Logging function for this strategy.
		:param txt:
		:param dt:
		:return:
		"""
		dt = dt or self.datas[0].datetime.date(0)
		print(f"{dt.isoformat()}, {txt}")

	def __init__(self):
		"""
		Keeps a reference to the "close" line in the data[0] dataseries
		"""
		self.dataclose = self.datas[0].close
		self.order = None  # No order yet

	# Already in base class method
	def notify_order(self, order):
		if order.status in [order.Submitted, order.Accepted]:
			return
		if order.status in [order.Completed]:
			if order.isbuy():
				self.log(f"BUY EXECUTED: {order.executed.price}")
			elif order.issell():
				self.log(f"SELL EXECUTED: {order.executed.price}")
			self.bar_executed = len(self)
		self.order = None

	def next(self):
		"""
		Specify the logic for the next bar.
		:return:
		"""
		if self.order:
			return

		if not self.position:
			if self.dataclose[0] < self.dataclose[-1]:
				if self.dataclose[-1] < self.dataclose[-2]:
					self.log(f"BUY CREATED: {self.dataclose[0]:.2f}")
					self.order = self.buy()  # Buy at market

		else:
			if len(self) >= (self.bar_executed + 5):
				self.log(f"SELL CREATED: {self.dataclose[0]}")
				self.order = self.sell()


if __name__ == "__main__":
	cerebro = bt.Cerebro()

	# Create a data feed
	start = datetime(2013, 1, 1)
	end = datetime.now()
	ticker = "CBA.AX"
	data = bt.feeds.PandasData(
		dataname=get_data(ticker, start, end)
	)

	# Add a strategy
	cerebro.addstrategy(TestStrategy)

	# Add data to the engine
	cerebro.adddata(data)

	# Add more stocks per order
	cerebro.addsizer(bt.sizers.PercentSizer, percents=70)

	# Set initial cash
	cerebro.broker.setcash(10000.0)

	print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

	cerebro.run()

	print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

	cerebro.plot()
