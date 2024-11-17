import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Lista plików z danymi i etykiety warunków
file_paths = ["amelka_uwu_ciekawe.csv", "amelka_uwu_nuda.csv", "amelka_uwu_ściana.csv"]
labels = ["ciekawe", "nudne", "baseline"]

# Parametry analizy
fs = 256  # Częstotliwość próbkowania
alpha_lowcut = 8.0
alpha_highcut = 12.0
beta_lowcut = 13.0
beta_highcut = 30.0
channels = [f"ch_{i}" for i in range(1, 9)]

# Funkcja filtru pasmowego
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Funkcja analizy dla jednego pliku
def analyze_data(file_path):
    data = pd.read_csv(file_path)
    data = data.dropna()

    # Przycięcie danych
    total_rows = len(data)
    if total_rows > 40000:
        data = data.iloc[20000:-20000]
    else:
        print(f"Za mało danych w pliku {file_path}!")
        return None, None
    
     # Usunięcie outlierów (wartości powyżej 100 µV lub poniżej -100 µV)
    for channel in channels:
        data[channel] = np.where((data[channel] > 100) | (data[channel] < -100), np.nan, data[channel])

    # Usunięcie brakujących wartości po usunięciu outlierów
    data = data.dropna()

    # Uśrednianie sygnałów ze wszystkich kanałów
    average_signal = data[channels].mean(axis=1)

    # Filtrowanie pasmowe
    filtered_alpha = bandpass_filter(average_signal, alpha_lowcut, alpha_highcut, fs)
    filtered_beta = bandpass_filter(average_signal, beta_lowcut, beta_highcut, fs)

    # Analiza widmowa (FFT)
    frequencies = np.fft.rfftfreq(len(filtered_alpha), 1/fs)
    fft_values_alpha = np.abs(np.fft.rfft(filtered_alpha))
    fft_values_beta = np.abs(np.fft.rfft(filtered_beta))

    return (filtered_alpha, filtered_beta), (frequencies, fft_values_alpha, fft_values_beta)

# Przechowywanie wyników analizy
signals = []
spectra = []

# Analiza dla każdego zbioru danych
for file_path in file_paths:
    signal, spectrum = analyze_data(file_path)
    if signal is not None and spectrum is not None:
        signals.append(signal)
        spectra.append(spectrum)

# Wizualizacja filtrowanych sygnałów (porównanie pasma alfa)
plt.figure(figsize=(12, 6))
for i, (filtered_alpha, filtered_beta) in enumerate(signals):
    plt.plot(filtered_alpha, label=f'Alfa - {labels[i]}', alpha = 0.15)
plt.xlabel("Próbka")
plt.ylabel("Amplituda")
plt.title("Porównanie filtrowanego sygnału EEG (Pasmo Alfa)")
plt.legend()
plt.grid()
plt.show()

# Wizualizacja filtrowanych sygnałów (porównanie pasma beta)
plt.figure(figsize=(12, 6))
for i, (filtered_alpha, filtered_beta) in enumerate(signals):
    plt.plot(filtered_beta, label=f'Beta - {labels[i]}', alpha = 0.15)
plt.xlabel("Próbka")
plt.ylabel("Amplituda")
plt.title("Porównanie filtrowanego sygnału EEG (Pasmo Beta)")
plt.legend()
plt.grid()
plt.show()

# Porównanie widm częstotliwościowych (alfa)
plt.figure(figsize=(12, 6))
for i, (frequencies, fft_values_alpha, fft_values_beta) in enumerate(spectra):
    plt.plot(frequencies, fft_values_alpha, label=f'Spectrum Alfa - {labels[i]}', alpha = 0.15)
plt.xlim(0, 30)
plt.xlabel("Częstotliwość (Hz)")
plt.ylabel("Amplituda")
plt.title("Porównanie widma częstotliwościowego (Pasmo Alfa)")
plt.legend()
plt.grid()
plt.show()

# Porównanie widm częstotliwościowych (beta)
plt.figure(figsize=(12, 6))
for i, (frequencies, fft_values_alpha, fft_values_beta) in enumerate(spectra):
    plt.plot(frequencies, fft_values_beta, label=f'Spectrum Beta - {labels[i]}', alpha = 0.15)
plt.xlim(0, 30)
plt.xlabel("Częstotliwość (Hz)")
plt.ylabel("Amplituda")
plt.title("Porównanie widma częstotliwościowego (Pasmo Beta)")
plt.legend()
plt.grid()
plt.show()
