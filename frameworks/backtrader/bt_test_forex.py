import backtrader as bt


class MyStrategy(bt.Strategy):
	def __init__(self):
		print("Initialising Strategy...")
		self.data_ready = False

	def notify_data(self, data, status, *args, **kwargs):
		print(f"Data status => {data._getstatusname(status)}")
		if status == data.LIVE:
			self.data_ready = True

	def log_data(self):
		ohlcv = [
			str(self.data.datetime.datetime()),
			str(self.data.open[0]),
			str(self.data.high[0]),
			str(self.data.low[0]),
			str(self.data.close[0]),
			str(self.data.volume[0])
		]
		print(", ".join(ohlcv))

	def next(self):
		if self.data_ready:
			self.log_data()
		# if not self.position:
		# 	self.buy()
		# elif self.position:
		# 	self.sell()


def start():
	"""
	Starts the Cerebro engine, and set up Interactive Brokers data feeds and
	strategy.

	:return:
	"""
	print("Starting Backtrader...")
	cerebro = bt.Cerebro()
	store = bt.stores.IBStore(port=7497, clientId=1)
	data = store.getdata(
		dataname="XAUUSD",
		sectype="CMDTY",
		exchange = "SMART",
		currency="USD",
		timeframe=bt.TimeFrame.Seconds
	)

	cerebro.resampledata(data, timeframe=bt.TimeFrame.Seconds, compression=15)

	# Resample secondly data into 15-second bars

	cerebro.broker = store.getbroker()
	cerebro.addstrategy(MyStrategy)
	cerebro.run()


if __name__ == "__main__":
	start()
