import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame, stripping any leading/trailing spaces from the column names
df = pd.read_csv('1-Flat_dim_960_batch_100000_k_4_latencies.csv')
df.columns = df.columns.str.strip()  # Strip any extra spaces from column names

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

    output_filename = '1-Flat_dim_960_batch_100000_k_4_latencies.png'
    plt.savefig(output_filename, format='png')
