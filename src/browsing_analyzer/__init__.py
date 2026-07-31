"""Browsing Analyzer - AI system for browsing history + RAM correlation analysis.

A modular, production-style package that:
1. Ingests Chrome browsing history and RAM usage logs (collected locally).
2. Preprocesses URLs, categorizes domains, and sessionizes browsing events.
3. Correlates browsing behavior with system/browser RAM usage.
4. Discovers behavior clusters (KMeans/GMM) and temporal patterns.
5. Trains a PyTorch LSTM to predict the next browsing category.
6. Generates traceable recommendations and interactive reports.
"""

__version__ = "1.0.0"
