import pandas as pd
import matplotlib.pyplot as plt
import re
import sys
import os

def load_csv(csv_filename):
    df = pd.read_csv(csv_filename, header=None)
    return df

def plot_df(df, output_filename, info, dim, nq):
    df.rename(columns={0: 'nb', df.columns[-1]: 'Avg'}, inplace=True)

    if 'Avg' not in df.columns:
        print("Error: 'Avg' column not found in the CSV file.")
        print("Available columns:", df.columns)
    else:
        # Extract nq values and average latencies
        nb_values = df['nb']
        avg_latencies = df['Avg']

        plt.figure(figsize=(10, 6))
        plt.plot(nb_values, avg_latencies, marker='o', linestyle='-', color='b')

        plt.xlabel('nb')
        plt.ylabel('Average Latency (µs)')
        plt.title(f'{info} Average Latency vs. nb with dim={dim}, nq={nq}')
        plt.grid(True)
        plt.ylim(bottom=0)
        # Save the plot to a file, incorporating dim and batch into the file name
        plt.savefig(output_filename, format='pdf')
        plt.show()

if __name__ == "__main__":
    arguments = sys.argv
    if len(sys.argv) < 3:
        print("Usage: python3 visualize_latency_nb.py <csv_filename> <output_dir>")
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

    # Extract information before '_dim'
    match_info = re.search(r'(.+)_dim_(\d+)_nq_(\d+)', csv_filename)
    if match_info:
        info = match_info.group(1)  # Info before '_dim'
        dim = match_info.group(2)
        nq = match_info.group(3)
    else:
        print("Error: Could not parse 'info', 'dim', and 'nb' from the file name.")
        info = "unknown"
        dim = "unknown"
        nq = "unknown"

    df = load_csv(csv_filename)
    print(output_pathname)
    plot_df(df, output_pathname, info, dim, nq)