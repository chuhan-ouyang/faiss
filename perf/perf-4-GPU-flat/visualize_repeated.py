import pandas as pd
import matplotlib.pyplot as plt
import re

# CSV file name
csv_filename = '4-GPU-flat-rep_dim_128_nb_10000_k_4_latencies.csv'

# Extract dim and nb from the file name using regex
match = re.search(r'dim_(\d+)_nb_(\d+)', csv_filename)
if match:
    dim = match.group(1)
    nb = match.group(2)
else:
    print("Error: Could not parse 'dim' and 'nb' from the file name.")
    dim = "unknown"
    nb = "unknown"

# Load the CSV file into a DataFrame (single row of latencies)
df = pd.read_csv(csv_filename, header=None)

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
output_filename = f'iteration_latency_plot_dim_{dim}_nb_{nb}.png'
plt.savefig(output_filename, format='png')
plt.show()  # Display the plot if running interactively