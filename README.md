# Event-driven Algorithmic Trading System written in Python

**THIS REPOSITORY DOES NOT CONSTITUTE FINANCIAL ADVICE. IT IS SOLELY FOR ENTERTAINMENT AND EDUCATIONAL PURPOSES. PLEASE EXERCISE CAUTION IN USING THIS SOFTWARE, AS I WILL NOT BE RESPONSIBLE FOR ANY OF YOUR GAINS/LOSSES IN USING THIS SOFTWARE. **

This repository is dedicated to storing different code pieces of my hobby project on algorithmic trading using Python.

# Project Structure
This repository has a few main directories:

- The `pytrading` directory contains the source code for the event-driven system, which is based on the Advanced Algorithmic Trading book by Michael Halls-Moore (https://www.quantstart.com/advanced-algorithmic-trading-ebook/). This is a very simple system, yet very scalable as the main logic is complete. All you need to do is add a strategy in `pytrading/strategy` and then run that same script in the terminal.

- The `frameworks/freqtrade` directory contains various trading strategies, both free from the Internet and from my own developing. These strategies use Freqtrade as the open-source backend bot to run. Feel free to experiment on any of these models.

- The `frameworks/backtrader` directory contains a few strategies using the open-source `backtrader` framework. 
