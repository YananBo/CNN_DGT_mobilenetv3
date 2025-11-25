import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_file(file_path):
    """
    Plots the iteration vs accuracy for a specific CSV file.

    Args:
        file_path (str): Path to the CSV file.
    """
    if not os.path.isfile(file_path):
        print(f"File {file_path} does not exist.")
        return

    data = pd.read_csv(file_path)

    # Check if the required columns exist
    if 'iteration' in data.columns and 'accuracy' in data.columns:
        plt.plot(data['iteration'], data['accuracy'], label=os.path.basename(file_path))
    else:
        print(f"Skipping {file_path}: Required columns 'iteration' and 'accuracy' not found.")
        return

    # Add labels, legend, and title
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Iteration vs Accuracy')
    plt.legend()

    # Show the plot
    plt.show()

if __name__ == "__main__":
    file_name = input("Enter the name of the CSV file to plot: ")
    results_dir = "results"  # Directory containing the results
    file_path = os.path.join(results_dir, file_name)
    plot_file(file_path)
