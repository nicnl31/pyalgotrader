#!/usr/bin/python
# -*- coding: utf-8 -*-

# session.py

from abc import ABCMeta, abstractmethod


class TradingSession(object):
	"""
	Base interface to a backtested or live trading session.
	"""
	__metaclass__ = ABCMeta

	@abstractmethod
	def run(self):
		raise NotImplementedError("Should implement run()")
