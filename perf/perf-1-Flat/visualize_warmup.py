import pandas as pd
import matplotlib.pyplot as plt
import re

# CSV file name
csv_filename = '1-Flat_warmup_dim_960_batch_500000_k_4_latencies.csv'

# Extract dim and batch from the file name using regex
match = re.search(r'dim_(\d+)_batch_(\d+)', csv_filename)
if match:
    dim = match.group(1)
    batch = match.group(2)
else:
    print("Error: Could not parse 'dim' and 'batch' from the file name.")
    dim = "unknown"
    batch = "unknown"

# Load the CSV file into a DataFrame (single row of latencies)
df = pd.read_csv(csv_filename, header=None)

# Extract the row of latencies as a list
latencies = df.iloc[0].values

# Generate iteration numbers (0, 1, 2, ...) based on the length of latencies
iterations = list(range(len(latencies)))

# Plot the data using a line chart
plt.figure(figsize=(10, 6))
plt.plot(iterations, latencies, marker='o', linestyle='-', color='b')

# Add labels and title with dim and batch in the title
plt.xlabel('Iteration')
plt.ylabel('Latency (µs)')
plt.title(f'Latency per Iteration with dim={dim}, nb={batch}')
plt.grid(True)

# Ensure y-axis starts from zero
plt.ylim(bottom=0)

# Save the plot to a file, incorporating dim and batch into the file name
output_filename = f'iteration_latency_plot_dim_{dim}_batch_{batch}.png'
plt.savefig(output_filename, format='png')
plt.show()  # Display the plot if running interactively