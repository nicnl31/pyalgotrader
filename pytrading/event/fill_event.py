#!/usr/bin/python
# -*- coding: utf-8 -*-

# fill_event.py

from __future__ import print_function
from pytrading.event.event import Event


class FillEvent(Event):
    """
    Encapsulates the notion of a Filled Order, the attributes of which
    are as returned from the brokerage. Stores the quantity of an
    instrument actually filled, and at what price. In addition, stores
    the commission of the trade from the brokerage.
    """
    def __init__(
            self,
            time_index,
            symbol,
            exchange,
            quantity,
            direction,
            fill_cost,
            commission=None
    ):
        """
        Initialises the FillEvent object. Sets the symbol, exchange,
        quantity, direction, cost of fill, and an optional commission.

        If commission is not provided, the Fill object will calculate
        it based on the trade size and fees as seen on Interactive Brokers.

        :param time_index:
            The bar resolution when the order was filled
        :param symbol:
            The instrument that was filled
        :param exchange:
            The exchange where the order was filled
        :param quantity:
            The filled quantity
        :param direction:
            The direction of fill ("BUY" or "SELL")
        :param fill_cost:
            The holdings' values, in dollars
        :param commission:
            An optional commission sent from Interactive Brokers
        """
        self.type = "FILL"
        self.time_index = time_index
        self.symbol = symbol
        self.exchange = exchange
        self.quantity = quantity
        self.direction = direction
        self.fill_cost = fill_cost
        if commission is None:
            self.commission = self.calculate_commission()
        else:
            self.commission = commission

    def calculate_commission(self, tier="I"):
        """
        Calculates the commission, as seen in Interactive Brokers.
            - Prices are in USD
            - Inclusive of GST
            - Not inclusive of third party fees
        Reference: https://www.interactivebrokers.com.au/en/pricing/commissions-stocks-asia-pacific.php?re=apac

        :param market: The market for commission structure
        :param tier: customer tier
        :return:
        """
        min_per_order = 0.35
        tiers = {
            "I": max(self.quantity*0.0035, min_per_order),
            "II": max(self.quantity*0.0020, min_per_order),
            "III": max(self.quantity*0.0015, min_per_order),
            "IV": max(self.quantity*0.0010, min_per_order),
            "V": max(self.quantity*0.0005, min_per_order)
        }
        return tiers[tier]
