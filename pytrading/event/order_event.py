#!/usr/bin/python
# -*- coding: utf-8 -*-

# order_event.py

from __future__ import print_function
from pytrading.event.event import Event


class OrderEvent(Event):
    """
    Handles the event of sending an Order to an execution system. The order
    contains:
        - A symbol (e.g. "GOOG")
        - A type ("MKT", "LMT", "STP", "BRACKET"). Bracket orders have Limit Buy
          ("LMT"), Profit Taker Sell ("LMT"), and Stop Loss Sell ("STP").
        - Quantity
        - Direction ("BUY" or "SELL")

    Explanation:
    When a Portfolio object receives SignalEvents, it assesses them in the
    context of the portfolio, in terms of risk and position sizing. This
    leads to OrderEvents that will be sent to an Execution Handler.
    """
    def __init__(
            self,
            symbol,
            order_type,
            quantity,
            direction
    ):
        """
        Initialises the OrderEvent, which includes:
        :param symbol:
            The symbol to trade
        :param order_type:
            A Market ("MKT") order or a Limit ("LMT") order
        :param quantity:
            The quantity of shares (integer, or fractional if allowed)
        :param direction:
            "BUY" (long) or "SELL" (short)
        """
        self.type = "ORDER"
        self.symbol = symbol
        self.order_type = order_type
        self.quantity = quantity
        self.direction = direction

    def print_order(self):
        """
        Prints the values within the order.
        :return: None.
        """
        print(
            f"ORDER: Symbol={self.symbol}, Type={self.order_type}, Quantity=\
{self.quantity}, Direction={self.direction}"
        )
