"""
Synthetic Time Series Data Generator for Anomaly Detection
Author: Vaishnav M
Date: November 2025

This module generates realistic synthetic time series data with embedded anomalies
for testing and development of anomaly detection models.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Dict, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """
    Generate synthetic time series data that simulates IoT sensor readings
    from manufacturing equipment.
    
    The data includes:
    - Multiple sensor channels (temperature, vibration, pressure, etc.)
    - Realistic patterns: trends, seasonality, noise
    - Embedded anomalies: spikes, drops, gradual drifts
    - Labels for supervised evaluation
    """
    
    def __init__(self, 
                 n_samples: int = 10000,
                 n_sensors: int = 4,
                 anomaly_ratio: float = 0.05,
                 random_seed: int = 42):
        """
        Initialize the synthetic data generator.
        
        Parameters:
        -----------
        n_samples : int
            Total number of time points to generate
        n_sensors : int
            Number of sensor channels (features)
        anomaly_ratio : float
            Proportion of data points that are anomalies (0.0 to 1.0)
        random_seed : int
            Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.n_sensors = n_sensors
        self.anomaly_ratio = anomaly_ratio
        self.random_seed = random_seed
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        logger.info(f"Initialized SyntheticDataGenerator:")
        logger.info(f"  - Samples: {n_samples}")
        logger.info(f"  - Sensors: {n_sensors}")
        logger.info(f"  - Anomaly Ratio: {anomaly_ratio*100:.1f}%")
    
    def generate_timestamps(self, 
                          start_date: str = '2024-01-01',
                          freq: str = '1min') -> pd.DatetimeIndex:
        """
        Generate timestamps for the time series.
        
        Parameters:
        -----------
        start_date : str
            Starting date for the time series
        freq : str
            Frequency of measurements (e.g., '1min', '5min', '1H')
            
        Returns:
        --------
        pd.DatetimeIndex
            Array of timestamps
        """
        timestamps = pd.date_range(
            start=start_date,
            periods=self.n_samples,
            freq=freq
        )
        logger.info(f"Generated timestamps from {timestamps[0]} to {timestamps[-1]}")
        return timestamps
    
    def generate_normal_pattern(self, 
                               timestamps: pd.DatetimeIndex,
                               base_value: float = 50.0,
                               trend_coefficient: float = 0.001,
                               seasonal_amplitude: float = 10.0,
                               noise_level: float = 2.0) -> np.ndarray:
        """
        Generate normal (non-anomalous) time series pattern.
        
        This simulates typical sensor behavior with:
        1. Base value (mean)
        2. Linear trend (gradual increase/decrease over time)
        3. Seasonal pattern (daily cycles)
        4. Random noise (measurement uncertainty)
        
        Parameters:
        -----------
        timestamps : pd.DatetimeIndex
            Time points
        base_value : float
            Base level of the sensor readings
        trend_coefficient : float
            Slope of linear trend (positive = increasing)
        seasonal_amplitude : float
            Amplitude of seasonal (daily) pattern
        noise_level : float
            Standard deviation of Gaussian noise
            
        Returns:
        --------
        np.ndarray
            Normal time series values
        """
        n = len(timestamps)
        
        # Component 1: Base value
        base = np.ones(n) * base_value
        
        # Component 2: Linear trend
        # Gradual increase or decrease over time
        trend = trend_coefficient * np.arange(n)
        
        # Component 3: Seasonal pattern (daily cycle)
        # Extract hour of day to create 24-hour periodicity
        hours = timestamps.hour + timestamps.minute / 60.0
        seasonal = seasonal_amplitude * np.sin(2 * np.pi * hours / 24)
        
        # Component 4: Random noise
        # Simulates measurement uncertainty and small fluctuations
        noise = np.random.normal(0, noise_level, n)
        
        # Combine all components
        signal = base + trend + seasonal + noise
        
        return signal
    
    def inject_point_anomalies(self,
                               data: np.ndarray,
                               n_anomalies: int,
                               anomaly_type: str = 'spike') -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject point anomalies (sudden spikes or drops).
        
        Point anomalies are individual data points that deviate significantly
        from the normal pattern. Examples: sensor glitch, sudden shock.
        
        Parameters:
        -----------
        data : np.ndarray
            Normal time series data
        n_anomalies : int
            Number of point anomalies to inject
        anomaly_type : str
            Type of anomaly: 'spike' (sudden increase) or 'drop' (sudden decrease)
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Modified data and binary labels (1 = anomaly, 0 = normal)
        """
        # Ensure data is numpy array
        data_copy = np.array(data).copy()
        labels = np.zeros(len(data), dtype=int)
        
        # Randomly select positions for anomalies
        # Avoid edges to prevent boundary effects
        anomaly_indices = np.random.choice(
            range(100, len(data) - 100),
            size=n_anomalies,
            replace=False
        )
        
        for idx in anomaly_indices:
            if anomaly_type == 'spike':
                # Spike: multiply by 2-4x
                data_copy[idx] *= np.random.uniform(2.0, 4.0)
            elif anomaly_type == 'drop':
                # Drop: multiply by 0.2-0.5x
                data_copy[idx] *= np.random.uniform(0.2, 0.5)
            
            labels[idx] = 1
        
        logger.info(f"Injected {n_anomalies} {anomaly_type} anomalies")
        return data_copy, labels
    
    def inject_contextual_anomalies(self,
                                    data: np.ndarray,
                                    n_anomalies: int,
                                    window_size: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject contextual anomalies (unusual patterns in specific context).
        
        Contextual anomalies appear normal in isolation but are anomalous
        in their context. Example: high temperature at night when it should be low.
        
        Parameters:
        -----------
        data : np.ndarray
            Normal time series data
        n_anomalies : int
            Number of contextual anomalies to inject
        window_size : int
            Duration of anomalous behavior (number of points)
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Modified data and binary labels
        """
        # Ensure data is numpy array
        data_copy = np.array(data).copy()
        labels = np.zeros(len(data), dtype=int)
        
        for _ in range(n_anomalies):
            # Random start position
            start_idx = np.random.randint(100, len(data) - window_size - 100)
            end_idx = start_idx + window_size
            
            # Invert the seasonal pattern in this window
            # This creates a "wrong time of day" anomaly
            mean_val = np.mean(data_copy[start_idx:end_idx])
            data_copy[start_idx:end_idx] = 2 * mean_val - data_copy[start_idx:end_idx]
            
            labels[start_idx:end_idx] = 1
        
        logger.info(f"Injected {n_anomalies} contextual anomalies")
        return data_copy, labels
    
    def inject_collective_anomalies(self,
                                    data: np.ndarray,
                                    n_anomalies: int,
                                    window_size: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject collective anomalies (gradual drift or sustained change).
        
        Collective anomalies are sequences of points that together form
        an anomalous pattern. Example: gradual bearing wear, sustained overheating.
        
        Parameters:
        -----------
        data : np.ndarray
            Normal time series data
        n_anomalies : int
            Number of collective anomalies to inject
        window_size : int
            Duration of anomalous behavior
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Modified data and binary labels
        """
        # Ensure data is numpy array
        data_copy = np.array(data).copy()
        labels = np.zeros(len(data), dtype=int)
        
        for _ in range(n_anomalies):
            start_idx = np.random.randint(100, len(data) - window_size - 100)
            end_idx = start_idx + window_size
            
            # Add gradual drift
            drift = np.linspace(0, np.random.uniform(10, 30), window_size)
            data_copy[start_idx:end_idx] += drift
            
            labels[start_idx:end_idx] = 1
        
        logger.info(f"Injected {n_anomalies} collective anomalies")
        return data_copy, labels
    
    def generate_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate complete synthetic dataset with multiple sensors and anomalies.
        
        Returns:
        --------
        Tuple[pd.DataFrame, pd.Series]
            - DataFrame with timestamps and sensor readings
            - Series with binary labels (1 = anomaly, 0 = normal)
        """
        logger.info("Starting dataset generation...")
        
        # Generate timestamps
        timestamps = self.generate_timestamps()
        
        # Initialize data dictionary
        data_dict = {'timestamp': timestamps}
        
        # Calculate number of anomalies per type
        total_anomalies = int(self.n_samples * self.anomaly_ratio)
        n_point = total_anomalies // 3
        n_contextual = total_anomalies // 3
        n_collective = total_anomalies - n_point - n_contextual
        
        # Master labels (will be OR of all sensor labels)
        master_labels = np.zeros(self.n_samples, dtype=int)
        
        # Generate data for each sensor
        for sensor_idx in range(self.n_sensors):
            logger.info(f"Generating sensor {sensor_idx + 1}/{self.n_sensors}...")
            
            # Generate normal pattern with different characteristics per sensor
            base_value = np.random.uniform(30, 70)
            trend = np.random.uniform(-0.002, 0.002)
            seasonal_amp = np.random.uniform(5, 15)
            noise = np.random.uniform(1, 3)
            
            normal_data = self.generate_normal_pattern(
                timestamps,
                base_value=base_value,
                trend_coefficient=trend,
                seasonal_amplitude=seasonal_amp,
                noise_level=noise
            )
            
            # Inject different types of anomalies more realistically
            # Distribute anomalies across sensors to minimize overlap
            sensor_labels = np.zeros(self.n_samples, dtype=int)
            data_with_anomalies = normal_data.copy()
            
            # Each sensor gets only ONE type of anomaly to reduce overlap
            # Reduce counts significantly since contextual/collective affect multiple points
            if sensor_idx == 0:
                # Sensor 1: Point anomalies (spikes only) - 40 single-point spikes
                data_with_anomalies, labels = self.inject_point_anomalies(
                    data_with_anomalies, 40, anomaly_type='spike'
                )
                sensor_labels = np.logical_or(sensor_labels, labels).astype(int)
                logger.info(f"Injected {np.sum(labels)} spike anomalies")
            
            elif sensor_idx == 1:
                # Sensor 2: Point anomalies (drops only) - 40 single-point drops
                data_with_anomalies, labels = self.inject_point_anomalies(
                    data_with_anomalies, 40, anomaly_type='drop'
                )
                sensor_labels = np.logical_or(sensor_labels, labels).astype(int)
                logger.info(f"Injected {np.sum(labels)} drop anomalies")
            
            elif sensor_idx == 2:
                # Sensor 3: Contextual anomalies only - 3 windows (each ~50 points)
                data_with_anomalies, labels = self.inject_contextual_anomalies(
                    data_with_anomalies, 3
                )
                sensor_labels = np.logical_or(sensor_labels, labels).astype(int)
                logger.info(f"Injected {np.sum(labels)} contextual anomaly points")
            
            elif sensor_idx == 3:
                # Sensor 4: Collective anomalies only - 2 windows (each ~100 points)
                data_with_anomalies, labels = self.inject_collective_anomalies(
                    data_with_anomalies, 2
                )
                sensor_labels = np.logical_or(sensor_labels, labels).astype(int)
                logger.info(f"Injected {np.sum(labels)} collective anomaly points")
            
            # Add to data dictionary
            sensor_name = f'sensor_{sensor_idx + 1}'
            data_dict[sensor_name] = data_with_anomalies
            
            # Update master labels (anomaly in ANY sensor = anomaly)
            master_labels = np.logical_or(master_labels, sensor_labels).astype(int)
        
        # Create DataFrame
        df = pd.DataFrame(data_dict)
        labels_series = pd.Series(master_labels, name='anomaly')
        
        # Summary statistics
        n_anomalies = labels_series.sum()
        anomaly_pct = (n_anomalies / len(labels_series)) * 100
        
        logger.info("=" * 60)
        logger.info("Dataset generation complete!")
        logger.info(f"Total samples: {len(df)}")
        logger.info(f"Total anomalies: {n_anomalies} ({anomaly_pct:.2f}%)")
        logger.info(f"Normal samples: {len(df) - n_anomalies}")
        logger.info(f"Sensor channels: {self.n_sensors}")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info("=" * 60)
        
        return df, labels_series
    
    def save_dataset(self, 
                    df: pd.DataFrame, 
                    labels: pd.Series,
                    output_dir: str = 'data/raw') -> None:
        """
        Save generated dataset to CSV files.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset with timestamps and sensor readings
        labels : pd.Series
            Binary anomaly labels
        output_dir : str
            Directory to save files
        """
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save data
        data_path = os.path.join(output_dir, 'synthetic_sensor_data.csv')
        df.to_csv(data_path, index=False)
        logger.info(f"Saved data to: {data_path}")
        
        # Save labels
        labels_path = os.path.join(output_dir, 'synthetic_sensor_labels.csv')
        labels.to_csv(labels_path, index=False, header=True)
        logger.info(f"Saved labels to: {labels_path}")
        
        # Save combined (for convenience)
        combined = df.copy()
        combined['anomaly'] = labels
        combined_path = os.path.join(output_dir, 'synthetic_sensor_data_with_labels.csv')
        combined.to_csv(combined_path, index=False)
        logger.info(f"Saved combined data to: {combined_path}")


def main():
    """
    Main function to generate and save synthetic dataset.
    """
    print("\n" + "=" * 60)
    print("SYNTHETIC TIME SERIES DATA GENERATOR")
    print("=" * 60 + "\n")
    
    # Create generator with realistic anomaly ratio
    generator = SyntheticDataGenerator(
        n_samples=10000,      # 10,000 time points (~1 week at 1min intervals)
        n_sensors=4,          # 4 sensor channels
        anomaly_ratio=0.015,  # 1.5% anomalies (realistic for IoT systems)
        random_seed=42
    )
    
    # Generate dataset
    df, labels = generator.generate_dataset()
    
    # Save to disk
    generator.save_dataset(df, labels)
    
    print("\n" + "=" * 60)
    print("✓ Dataset generation complete!")
    print("=" * 60 + "\n")
    
    # Display sample
    print("First 5 rows:")
    print(df.head())
    print("\nLabel distribution:")
    print(f"Normal (0): {(labels == 0).sum()}")
    print(f"Anomaly (1): {(labels == 1).sum()}")


if __name__ == "__main__":
    main()
