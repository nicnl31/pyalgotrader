#!/usr/bin/python
# -*- coding: utf-8 -*-

# market_event.py

from __future__ import print_function
from pytrading.event.event import Event


class MarketEvent(Event):
    """
    Handles the Event of receiving a new market update from the brokerage
    with corresponding bars. Each bar corresponds to a trading period.
    """
    def __init__(self):
        self.type = "MARKET"
