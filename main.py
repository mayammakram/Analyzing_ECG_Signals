import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def FivePointDiff(input_file, sample_rate):
    x = np.loadtxt(input_file)
    T = 1 / sample_rate
    y = np.zeros_like(x)

    for n in range(2, len(x) - 2):
        y[n] = (1 / (8 * T)) * (-x[n - 2] - 2 * x[n - 1] + 2 * x[n + 1] + x[n + 2]) ** 2
    
    return y

def smoothing(x, N):
    # Initialize output array
    y = np.zeros_like(x)

    # Compute initial average
    y[N-1] = np.mean(x[:N])

    # Apply recursive moving average equation
    for n in range(N, len(x)):
        y[n] = y[n-1] + (1/N)*(x[n] - x[n-N])

    return y

def autocorrelation(y):
    n = len(y)
    A = np.zeros(n)
    for m in range(n):
        for i in range(m, n):
            A[m] += y[i] * y[i-m]
    return A

def avg_time_between_beats(A, sample_rate):
    peaks, _ = find_peaks(A, height=np.percentile(A, 92)) 
    T = 1 / sample_rate
    beat_times = np.diff(peaks) * T
    avg_time = np.mean(beat_times) if len(beat_times) > 0 else 0

    return avg_time

def avg_heart_rate(A, sample_rate):
    avg_time = avg_time_between_beats(A, sample_rate)
    avg_rate = 60 / avg_time if avg_time > 0 else 0
    return avg_rate


#description:  compute_frequency takes the autocorrelation array A and
# the sample rate as arguments. It calculates the beat times between peaks, converts 
# them to frequencies by taking the reciprocal, and then calculates the average frequency.
def compute_frequency(A, sample_rate):
    peaks, _ = find_peaks(A, height=np.percentile(A, 92))  
    T = 1 / sample_rate
    beat_times = np.diff(peaks) * T
    frequencies = 1 / beat_times if len(beat_times) > 0 else []
    avg_frequency = np.mean(frequencies) if len(frequencies) > 0 else 0

    return avg_frequency


def plot_ecg_signal(x, dx2, y, A):
    # Create figure
    fig, axs = plt.subplots(4, 1)
    
    # Plot first 2000 samples of signal
    axs[0].plot(x[:2000])
    axs[0].set_title('Raw ECG Signal')
    axs[0].set_xlabel(' ')
    axs[0].set_ylabel('Amplitude')
    
    # Plot derivative of signal
    axs[1].plot(dx2[:2000])
    axs[1].set_title('Derivative Sqaured of ECG Signal')
    axs[1].set_xlabel(' ')
    axs[1].set_ylabel('Amplitude')
    
    # Plot smoothed signal
    axs[2].plot(y[:2000])
    axs[2].set_title('Smoothed ECG Signal')
    axs[2].set_xlabel(' ')
    axs[2].set_ylabel('Amplitude')
    
    # Plot autocorrelated signal
    axs[3].plot(A[:2000])
    axs[3].set_title('Autocorrelated ECG Signal')
    axs[3].set_xlabel(' ')
    axs[3].set_ylabel('Amplitude')

    # Show figure
    plt.show()


# Read data from .txt file
file = 'Data1.txt'
x = np.loadtxt(file)

sample_rate = 512
dx2 = FivePointDiff(file, sample_rate)
y = smoothing(dx2, 31)
A = autocorrelation(y)
plot_ecg_signal(x, dx2, y,A)

print("The Average Frequency is", compute_frequency(A, sample_rate))
print("The Average Time between Beats is", avg_time_between_beats(A, sample_rate))
print("The Average Heart Rate is", avg_heart_rate(A, sample_rate))
