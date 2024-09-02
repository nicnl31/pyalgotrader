#!/usr/bin/python
# -*- coding: utf-8 -*-

# event.py

###############################################################################
# The Event class hierarchy provides 4 types of Events which allow
# communication between the components of an event-driven system, via an event
# queue. They are a MarketEvent, SignalEvent, OrderEvent, and FillEvent.
###############################################################################

from __future__ import print_function


class Event(object):
    """
    Event is a base class, providing an interface for all subsequent
    (inherited) Events, that will trigger further Events in the
    trading infrastructure.
    """
    pass
