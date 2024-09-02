from __future__ import print_function

from abc import ABCMeta, abstractmethod


class Resolution(object):
	__metaclass__ = ABCMeta

	@abstractmethod
	def get_interval(self):
		raise NotImplementedError("Should implement get_interval()")

	@abstractmethod
	def get_resolution(self):
		raise NotImplementedError("Should implement get_resolution()")

	@abstractmethod
	def get_n_periods(self):
		raise NotImplementedError("Should implement get_n_periods()")


class Daily(Resolution):
	def __init__(self):
		self.resolution = "Daily"
		self.interval = "1d"
		self.periods = 252

	def get_interval(self):
		return self.interval

	def get_resolution(self):
		return self.resolution

	def get_n_periods(self):
		return self.periods


class Monthly(Resolution):
	def __init__(self):
		self.resolution = "Monthly"
		self.interval = "1mo"
		self.periods = 12

	def get_interval(self):
		return self.interval

	def get_resolution(self):
		return self.resolution

	def get_n_periods(self):
		return self.periods


class Hourly(Resolution):
	def __init__(self):
		self.resolution = "Hourly"
		self.interval = "1h"
		self.periods = 252*6.5

	def get_interval(self):
		return self.interval

	def get_resolution(self):
		return self.resolution

	def get_n_periods(self):
		return self.periods


class Minutely(Resolution):
	def __init__(self, n_minutes=1):
		self.n_minutes = n_minutes
		self.resolution = "Minutely"
		self.interval = f"{n_minutes}m"
		self.periods = 252 * 6.5 * (60/self.n_minutes)

	def get_interval(self):
		return self.interval

	def get_resolution(self):
		return self.resolution

	def get_n_periods(self):
		return self.periods
