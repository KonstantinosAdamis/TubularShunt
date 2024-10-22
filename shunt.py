import numpy as np
import matplotlib.pyplot as plt

# Define a time domain signal
Fs = 20000000  # Sampling frequency (Hz)
T = 1 / Fs  # Sampling interval
t = np.arange(0, 1, T)  # Time vector from 0 to 1 second

# Signal: h(t)
signal = 177187*np.exp(-8.85936*10**7*t)

# Compute DFT using numpy's FFT function
dft = np.fft.fft(signal)
N = len(dft)
frequencies = np.fft.fftfreq(N, T)  # Frequency vector

# Take only the positive frequencies and normalize the magnitude
positive_frequencies = frequencies[:N // 2]
magnitude = np.abs(dft[:N // 2]) / N

# Plot the time domain signal
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title("Time Domain Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

# Plot the DFT (frequency domain)
plt.subplot(2, 1, 2)
plt.plot(positive_frequencies, magnitude)
plt.title("Frequency Domain (DFT)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid()

plt.tight_layout()
plt.show()
