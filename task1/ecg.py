import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch
import wfdb

# --- Load the ECG Signal (Corrected Method) ---
# Each command is now on its own line
record = wfdb.rdrecord('100', pn_dir='mitdb', sampto=3600)
signal = record.p_signal[:, 0] # Use the first channel
fs = record.fs # Get the sampling frequency (should be 360 Hz)

# Create a time vector for plotting
time = np.arange(len(signal)) / fs

# --- Plot the Raw Signal ---
plt.figure(figsize=(12, 6))
plt.plot(time, signal)
plt.title("Raw, Noisy ECG Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.grid(True)
plt.show()








# --- 1. High-Pass Filter for Baseline Wander ---
cutoff_high = 0.5 # Hz
order = 2 # Filter order

# Design the Butterworth high-pass filter
b_high, a_high = butter(order, cutoff_high / (fs / 2), btype='high')

# Apply the filter
signal_hp = filtfilt(b_high, a_high, signal)

# --- Plot the result ---
plt.figure(figsize=(12, 6))
plt.plot(time, signal_hp, label='High-Pass Filtered')
plt.plot(time, signal, label='Raw Signal', alpha=0.5)
plt.title("ECG After Removing Baseline Wander")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.legend()
plt.grid(True)
plt.show()











# --- 2. Notch Filter for Powerline Interference ---
freq_notch = 50.0 # Hz (for Egypt)
Q = 30 # Quality factor, defines the filter's narrowness

# Design the IIR notch filter
b_notch, a_notch = iirnotch(freq_notch, Q, fs)

# Apply the filter to the high-pass filtered signal
signal_notch = filtfilt(b_notch, a_notch, signal_hp)

# --- Plot the result ---
plt.figure(figsize=(12, 6))
plt.plot(time, signal_notch)
plt.title("ECG After Notch Filter (50 Hz Removed)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.grid(True)
plt.show()











# --- 3. Low-Pass Filter for Muscle Artifacts ---
cutoff_low = 150.0 # Hz
order = 4

# Design the Butterworth low-pass filter
b_low, a_low = butter(order, cutoff_low / (fs / 2), btype='low')

# Apply the filter to the already filtered signal
signal_clean = filtfilt(b_low, a_low, signal_notch)

# --- Plot the Final Result ---
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(time, signal, label='Raw Signal')
plt.title("Comparison of Raw and Filtered ECG")
plt.ylabel("Amplitude (mV)")
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time, signal_clean, label='Fully Filtered Signal', color='red')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()