import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame without specifying column headers
df = pd.read_csv('4-GPU-flat_dim_128_batch_10000_k_4_latencies.csv', header=None)

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

    # Add labels and title
    plt.xlabel('Number of Queries (nq)')
    plt.ylabel('Average Latency (µs)')
    plt.title('Average Latency vs. Number of Queries')
    plt.grid(True)

    output_filename = '4-GPU-flat_dim_128_batch_10000_k_4_latencies.png'
    plt.savefig(output_filename, format='png')
    plt.show()  # Display the plot if running interactively