import pandas as pd
import matplotlib.pyplot as plt
import re

# CSV file name
csv_filename = '4-GPU-flat_200_5000_dim_128_batch_10000_k_4_latencies.csv'

# Extract dim and batch from the file name using regex
match = re.search(r'dim_(\d+)_batch_(\d+)', csv_filename)
if match:
    dim = match.group(1)
    batch = match.group(2)
else:
    print("Error: Could not parse 'dim' and 'batch' from the file name.")
    dim = "unknown"
    batch = "unknown"

# Load the CSV file into a DataFrame without specifying column headers
df = pd.read_csv(csv_filename, header=None)

# Assume the first column is 'nq' and the last column is 'Avg'
df.rename(columns={0: 'nq', df.columns[-1]: 'Avg'}, inplace=True)

# Check if 'Avg' column exists
if 'Avg' not in df.columns:
    print("Error: 'Avg' column not found in the CSV file.")
    print("Available columns:", df.columns)
else:
    # Extract nq values and average latencies
    nq_values = df['nq']
    avg_latencies = df['Avg']

    # Plot the data using a line chart
    plt.figure(figsize=(10, 6))
    plt.plot(nq_values, avg_latencies, marker='o', linestyle='-', color='b')

    # Add labels and title with dim and batch in the title
    plt.xlabel('Batch Size (nq)')
    plt.ylabel('Average Latency (µs)')
    plt.title(f'Average Latency vs. Batch Size with dim={dim}, nb={batch}')
    plt.grid(True)

    # Ensure y-axis starts from zero
    plt.ylim(bottom=0)

    # Save the plot to a file, incorporating dim and batch into the file name
    output_filename = f'latency_plot_dim_{dim}_batch_{batch}.png'
    plt.savefig(output_filename, format='png')
    plt.show()  # Display the plot if running interactively
