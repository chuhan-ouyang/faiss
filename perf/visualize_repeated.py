import pandas as pd
import matplotlib.pyplot as plt
import re
import sys
import os

def load_csv(csv_filename):
    # Load the CSV file into a DataFrame (single row of latencies)
    df = pd.read_csv(csv_filename, header=None)
    return df

def plot_df(df, output_filename, info, dim, nb):
    # Extract the row of latencies as a list
    latencies = df.iloc[0].values
    nq = latencies[0]  # First element is nq
    latencies = latencies[1:]  # Remaining elements are latencies

    # Generate iteration numbers (1, 2, 3, ...) based on the length of latencies
    iterations = list(range(1, len(latencies) + 1))

    # Plot the data using a line chart
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, latencies, marker='o', linestyle='-', color='b')

    # Add labels and title with dim, nb, nq, and info in the title
    plt.xlabel('Iteration')
    plt.ylabel('Latency (µs)')
    plt.title(f'{info} Latency per Iteration with dim={dim}, nb={nb}, nq={nq}')
    plt.grid(True)

    # Ensure y-axis starts from zero
    max_latency = max(latencies)
    plt.ylim(0, max_latency + 30)

    # Save the plot to a file, incorporating dim and nb into the file name
    plt.savefig(output_filename, format='pdf')
    plt.show()  # Display the plot if running interactively

if __name__ == "__main__":
    arguments = sys.argv
    if len(sys.argv) < 3:
        print("Usage: python3 visualize_repeated.py <csv_filename> <output_dir>")
        exit()
    csv_filename = sys.argv[1]
    output_dir = sys.argv[2]

    # Extract the portion before _dim as 'info'
    match_info = re.search(r'(.+)_dim_(\d+)_nb_(\d+)', csv_filename)
    if match_info:
        info = match_info.group(1)  # Part before '_dim'
        dim = match_info.group(2)
        nb = match_info.group(3)
    else:
        print("Error: Could not parse 'info', 'dim', and 'nb' from the file name.")
        info = "unknown"
        dim = "unknown"
        nb = "unknown"

    # Build output file path
    pos = csv_filename.rfind("/")
    if pos == -1:
        plot_file_name = csv_filename
    else:
        plot_file_name = csv_filename[pos+1:]
    output_pathcsv = os.path.join(output_dir, plot_file_name)
    output_pathname = output_pathcsv.replace(".csv", ".pdf")
    print(f"pdf_file_path: {output_pathname}")

    # Load CSV data
    df = load_csv(csv_filename)

    # Generate the plot
    plot_df(df, output_pathname, info, dim, nb)
