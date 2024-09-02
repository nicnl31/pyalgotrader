#!/usr/bin/python
# -*- coding: utf-8 -*-

# signal_event.py

from __future__ import print_function
from pytrading.event.event import Event


class SignalEvent(Event):
    """
    Handles the event of sending a Signal from a Strategy object. This is
    then received by a Portfolio object, as advice for how to trade the asset.
    """
    def __init__(
            self,
            strategy_id,
            symbol,
            datetime,
            signal_type,
            strength
    ):
        """
        Initialises the SignalEvent.

        :param strategy_id:
            the unique identifier for the strategy that generated the signal
        :param symbol:
            the ticker symbol, e.g. "GOOG"
        :param datetime:
            the timestamp at which the signal was generated
        :param signal_type:
            "LONG" or "SHORT" or "EXIT"
        :param strength:
            an adjustment factor "suggestion" used to scale quantity at the
            portfolio level. Useful for pairs trading strategies.
        """
        self.type = "SIGNAL"
        self.strategy_id = strategy_id
        self.symbol = symbol
        self.datetime = datetime
        self.signal_type = signal_type
        self.strength = strength
