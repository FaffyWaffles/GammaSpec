import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def load_and_normalize_histogram(file_path, normalization_method="total_counts"):
    # Load the CSV file
    data = pd.read_csv(file_path)

    # Convert the 'Time' column to datetime to calculate the duration
    data['Datetime'] = pd.to_datetime(data['Day'] + ' ' + data['Time'], format='%m/%d/%Y %H:%M:%S.%f')

    # Get the total counts per channel
    channel_counts = data['Channel#'].value_counts().sort_index()

    # Normalize the counts
    if normalization_method == "counts_per_second":
        duration_seconds = (data['Datetime'].max() - data['Datetime'].min()).total_seconds()
        normalized_counts = channel_counts / duration_seconds
    elif normalization_method == "total_counts":
        normalized_counts = channel_counts / channel_counts.sum()
    else:
        raise ValueError("Unsupported normalization method")

    return normalized_counts

def chi_squared_distance(hist1, hist2):
    # Make sure the histograms have the same length
    length = max(len(hist1), len(hist2))
    hist1 = np.pad(hist1, (0, length - len(hist1)), 'constant')
    hist2 = np.pad(hist2, (0, length - len(hist2)), 'constant')

    return np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + 1e-10))

def bhattacharyya_distance(hist1, hist2):
    # Make sure the histograms have the same length
    length = max(len(hist1), len(hist2))
    hist1 = np.pad(hist1, (0, length - len(hist1)), 'constant')
    hist2 = np.pad(hist2, (0, length - len(hist2)), 'constant')

    return -np.log(np.sum(np.sqrt(hist1 * hist2)))

def plot_histograms(unknown_histogram, closest_histogram, closest_isotope):
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.bar(unknown_histogram.index, unknown_histogram.values, color='blue', alpha=0.6, label='Unknown Sample')
    plt.title('Unknown Sample Histogram')
    plt.xlabel('Channel')
    plt.ylabel('Counts')
    plt.xlim(0, 1000)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.bar(closest_histogram.index, closest_histogram.values, color='green', alpha=0.6, label=f'Closest Match: {closest_isotope}')
    plt.title(f'Closest Match Histogram ({closest_isotope})')
    plt.xlabel('Channel')
    plt.ylabel('Counts')
    plt.xlim(0, 1000)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.bar(unknown_histogram.index, unknown_histogram.values, color='blue', alpha=0.6, label='Unknown Sample')
    plt.bar(closest_histogram.index, closest_histogram.values, color='green', alpha=0.4, label=f'Closest Match: {closest_isotope}')
    plt.title('Overlay of Unknown and Closest Match')
    plt.xlabel('Channel')
    plt.ylabel('Normalized Counts')
    plt.xlim(0, 1000)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # File paths for known isotopes
    known_isotopes = {
        "Ba133": 'C:/Projects/Python/GammaSpec/measurements/isotope_references/Ba133.csv',
        "Co57": 'C:/Projects/Python/GammaSpec/measurements/isotope_references/Co57.csv',
        "Co60": 'C:/Projects/Python/GammaSpec/measurements/isotope_references/Co60.csv',
        "Cs137": 'C:/Projects/Python/GammaSpec/measurements/isotope_references/Cs137.csv',
        "Na22": 'C:/Projects/Python/GammaSpec/measurements/isotope_references/Na22.csv'
    }

    # Load and normalize histograms for known isotopes
    known_histograms = {isotope: load_and_normalize_histogram(file_path) for isotope, file_path in known_isotopes.items()}

    # File path for the unknown sample
    unknown_file_path = 'C:/Projects/Python/GammaSpec/measurements/2024-07-10 Data Single source/1.A.04/Na22/2023-10-08-T1742 for 5.0min-Events.csv'

    # Load and normalize the histogram for the unknown sample
    unknown_histogram = load_and_normalize_histogram(unknown_file_path)

    # Calculate similarity measures
    chi2_distances = {}
    bhattacharyya_distances = {}

    for isotope, hist in known_histograms.items():
        chi2_distances[isotope] = chi_squared_distance(unknown_histogram.values, hist.values)
        bhattacharyya_distances[isotope] = bhattacharyya_distance(unknown_histogram.values, hist.values)

    # Find the closest match based on Chi-squared distance
    closest_match_chi2 = min(chi2_distances, key=chi2_distances.get)
    closest_match_bhattacharyya = min(bhattacharyya_distances, key=bhattacharyya_distances.get)

    print(f"Closest match based on Chi-squared distance: {closest_match_chi2} with distance {chi2_distances[closest_match_chi2]}")
    print(f"Closest match based on Bhattacharyya distance: {closest_match_bhattacharyya} with distance {bhattacharyya_distances[closest_match_bhattacharyya]}")

    # Plot the histograms of the unknown sample and the closest match
    plot_histograms(unknown_histogram, known_histograms[closest_match_chi2], closest_match_chi2)
