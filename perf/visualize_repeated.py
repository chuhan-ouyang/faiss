import pandas as pd
import matplotlib.pyplot as plt
import re
import sys
import os

def load_csv(csv_filename):
    # CSV file name
    # csv_filename = '4-GPU-flat-rep_dim_128_nb_10000_k_4_latencies.csv'
    # Load the CSV file into a DataFrame (single row of latencies)
    df = pd.read_csv(csv_filename, header=None)
    return df

def plot_df(df, output_filename, dim, nb):
    # Extract the row of latencies as a list
    latencies = df.iloc[0].values

    # Generate iteration numbers (0, 1, 2, ...) based on the length of latencies
    iterations = list(range(len(latencies)))

    # Plot the data using a line chart
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, latencies, marker='o', linestyle='-', color='b')

    # Add labels and title with dim and nb in the title
    plt.xlabel('Iteration')
    plt.ylabel('Latency (µs)')
    plt.title(f'Latency per Iteration with dim={dim}, nb={nb}')
    plt.grid(True)

    # Ensure y-axis starts from zero
    plt.ylim(bottom=0)

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
    pos = csv_filename.rfind("/")
    if pos == -1:
        plot_file_name = csv_filename
    else:
        plot_file_name = csv_filename[pos+1:]
    output_pathcsv = os.path.join(output_dir, plot_file_name)
    output_pathname = output_pathcsv.replace(".csv", ".pdf")

    match = re.search(r'dim_(\d+)_nb_(\d+)', csv_filename)
    if match:
        dim = match.group(1)
        nb = match.group(2)
    else:
        print("Error: Could not parse 'dim' and 'nb' from the file name.")
        dim = "unknown"
        nb = "unknown"
    
    df = load_csv(csv_filename)
    plot_df(df, output_pathname, dim, nb)
    
