import matplotlib.pyplot as plt
from scipy.signal import detrend, butter, filtfilt
import numpy as np
import random

class ResampleTrace:
    def __init__(self, downsample_factor):
        self.downsample_factor = downsample_factor
        
    def __name__(self):
        return "ResampleTrace"
    
    def __call__(self, data):
        return data[::self.downsample_factor]

class BandpassFilter:
    def __init__(self, lowcut, highcut, fs, order=5):
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order
        
    def __name__(self):
        return "BandpassFilter"

    def __call__(self, data):
        nyq = 0.5 * self.fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = butter(self.order, [low, high], btype='band')
        return filtfilt(b, a, detrend(data))

class AddNoise:
    def __init__(self, noise_level):
        self.noise_level = noise_level
        
    def __name__(self):
        return "AddNoise"
    
    def __call__(self, data):
        noise = np.random.normal(0, self.noise_level, data.shape)
        return data + noise

class LocalMinMaxNorm:
    
    def __name__(self):
        return "LocalMinMaxNorm"
    
    def __call__(self, data):
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val)

class AverageInterval:
    def __init__(self, max_length):
        self.max_length = max_length
    
    def __name__(self):
        return "AverageInterval"
    
    def __call__(self, data):
        start = np.random.randint(0, len(data) - self.max_length)
        interval_length = np.random.randint(1, self.max_length + 1)
        average_value = np.mean(data[start:start + interval_length])
        data[start:start + interval_length] = average_value
        return data

class Taper:
    def __init__(self, max_percentage=0.05):
        self.max_percentage = max_percentage
        
    def __name__(self):
        return "Taper"
    
    def __call__(self, data):
        taper_length = int(len(data) * self.max_percentage)
        window = np.hanning(taper_length * 2)
        data[:taper_length] *= window[:taper_length]
        data[-taper_length:] *= window[-taper_length:]
        return data
    
class GAFTransform:
    """Transform to compute Gramian Angular Field of a pre-normalized time series using NumPy."""
    def __name__(self):
        return "GAFTransform"
        
    def __call__(self, series):
        # Assume series is already normalized to [0, 1]
        # Convert normalized series to angles
        phi = np.arccos(series)  # No need to clip since series should already be in [0,1]

        # Create Gramian Angular Field
        gaf = np.cos(np.outer(phi, phi))
        return gaf
    
    
######################################## MFR Transforms below: ############################################
class FixedRangeNormalize:
    def __init__(self, min_val=3, max_val=60):
        self.min_val = min_val
        self.max_val = max_val
        
    def __name__(self):
        return "FixedRangeNormalize"

    def __call__(self, data):
        # Normalize data to range [0, 1]
        normalized_data = (data - self.min_val) / (self.max_val - self.min_val)
        # Ensure all data points are within [0, 1]
        normalized_data = np.clip(normalized_data, 0, 1)
        return normalized_data

class ApplyRandomColormap:
    def __init__(self, colormaps):
        # Store a list of colormap names
        self.colormaps = colormaps
        
    def __name__(self):
        return "ApplyRandomColormap"

    def __call__(self, data):
        # Randomly select a colormap
        colormap = random.choice(self.colormaps)
        cmap = plt.cm.get_cmap(colormap)
        # Apply the selected colormap to the normalized data
        colored_data = cmap(data)[:, :, :3]  # Convert to RGB
        return colored_data

