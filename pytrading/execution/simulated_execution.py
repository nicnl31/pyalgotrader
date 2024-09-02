#!/usr/bin/python
# -*- coding: utf-8 -*-

# simulated_execution.py

import datetime
from datetime import timezone
from queue import Queue
from pytrading.event.fill_event import FillEvent
from pytrading.event.order_event import OrderEvent
from pytrading.execution.execution import ExecutionHandler


class SimulatedExecutionHandler(ExecutionHandler):
	"""
	The simulated execution handler simply converts all Order objects into their
	equivalent Fill objects automatically, without latency, slippage or fill
	ratio issues.

	This allows a straightforward first-go test of any strategy, before
	implementation with a more sophisticated execution handler.
	"""
	def __init__(self, events: Queue):
		"""
		Initialises the handler, setting the events queue up internally.

		:param events: the events queue object
		"""
		self.events = events

	def execute_order(self, event: OrderEvent):
		"""
		Naively converts Order objects into Fill objects, i.e. without any
		latency, slippage, or fill ratio problems.

		:param event: The OrderEvent object with order information
		:return: None
		"""
		if event.type == "ORDER":
			fill_event = FillEvent(
				time_index=datetime.datetime.now(tz=timezone.utc),
				symbol=event.symbol,
				exchange="ARCA",  # ARCA is a placeholder only
				quantity=event.quantity,
				direction=event.direction,
				fill_cost=None  # Fill cost is already in the Portfolio object
			)
			self.events.put(fill_event)
