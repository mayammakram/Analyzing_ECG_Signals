import numpy as np
import matplotlib.pyplot as plt

def FivePointDiff(input_file, sample_rate):

    # Read data from .txt file
    x = np.loadtxt(input_file)
    
    T = 1/sample_rate
    
    # Initialize output array
    y = np.zeros_like(x)
    
    # Apply 5-point difference equation
    for n in range(2, len(x)-2):
        y[n] = (1/(8*T)) * (-x[n-2] - 2*x[n-1] + 2*x[n+1] + x[n+2])
    
    y_sq = np.square(y)
    
    return y_sq

def smoothing(x, N):
    # Initialize output array
    y = np.zeros_like(x)
    
    # Compute initial average
    y[N-1] = np.mean(x[:N])
    
    # Apply recursive moving average equation
    for n in range(N, len(x)):
        y[n] = y[n-1] + (1/N)*(x[n] - x[n-N])
    
    return y

# def Autocorrelation(lag, y):  
#     sum = 0
#     for i in range(len(y)):
#        sum += y[i]*y[i-lag]
    
#     return sum

#################
def autocorrelation(y):
    n = len(y)
    A = np.zeros(n)
    for m in range(n):
        for i in range(m, n):
            A[m] += y[i] * y[i-m]
        A[m] /= (n - m)
    return A

def avg_time_between_beats(y, sample_rate):
    # Compute autocorrelation
    A = autocorrelation(y)
    
    # Find first peak after 0 lag
    peak_index = np.argmax(A[1:]) + 1
    
    # Convert peak index to time
    T = 1/sample_rate
    avg_time = peak_index * T
    
    return avg_time



# def autocorrelation_plot(autocorrelation):

#     lag = len(autocorrelation)
#     plt.plot(lag, autocorrelation)
#     plt.xlabel('Lag')
#     plt.ylabel('Autocorrelation')
#     plt.title('Autocorrelation Plot')
#     plt.grid(True)
#     plt.show()


def plot_signal_and_autocorrelation(y, A):
    # Create figure
    fig, axs = plt.subplots(2, 1)
    
    # Plot first 2000 samples of signal
    axs[0].plot(y[:2000])
    axs[0].set_title('ECG Signal')
    axs[0].set_xlabel('Sample')
    axs[0].set_ylabel('Amplitude')
    
    # Plot autocorrelation
    axs[1].plot(A)
    axs[1].set_title('Autocorrelation')
    axs[1].set_xlabel('Lag')
    axs[1].set_ylabel('Autocorrelation')
    
    # Show figure
    plt.show()
    
def plot_ecg_signal(x, dx2, y):
    # Create figure
    fig, axs = plt.subplots(3, 1)
    
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
    
    # Show figure
    plt.show()


# Read data from .txt file
file = 'Data2.txt'
x = np.loadtxt(file)

sample_rate = 512
dx2 = FivePointDiff(file, sample_rate)
y = smoothing(dx2, 31)
A = autocorrelation(y)

plot_ecg_signal(x, dx2, y)


# plot_signal_and_autocorrelation(y, A)
print("The Average Time between Beats is", avg_time_between_beats(A, sample_rate))


