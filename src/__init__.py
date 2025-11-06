"""
Time Series Anomaly Detection Package
Author: Vaishnav M
Date: November 2025
"""

__version__ = "1.0.0"
__author__ = "Vaishnav M"

# Import main modules for easy access
from . import data_preparation
from . import feature_engineering
from . import evaluation
from . import visualization

__all__ = [
    'data_preparation',
    'feature_engineering', 
    'evaluation',
    'visualization'
]
