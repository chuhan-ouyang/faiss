import pandas as pd
import re
import sys
import os
import matplotlib.pyplot as plt

def load_csv(csv_filename):
    df = pd.read_csv(csv_filename, header=None)
    return df

def compute_variance(df):
    # Assuming the first column is 'nq' and the last column is 'Avg'
    data_columns = df.columns[1:-1]  # Exclude 'nq' and 'Avg' columns

    # Function to calculate variance for each row and print the values used
    def row_variance(row):
        data_points = row[data_columns].astype(float).values  # Extract values as float array
        print(f"Calculating variance for row {row.name}: values = {data_points}")  # Debug print
        return data_points.var()  # Calculate variance

    # Apply row-wise variance calculation with debug print
    df['Variance'] = df.apply(row_variance, axis=1)
    return df

def save_with_variance(df, original_filename, output_dir):
    pos = original_filename.rfind("/")
    if pos == -1:
        base_filename = original_filename
    else:
        base_filename = original_filename[pos+1:]
    new_filename = base_filename.replace(".csv", "_variance.csv")
    output_path = os.path.join(output_dir, new_filename)

    # Save the DataFrame with the new variance column
    df.to_csv(output_path, index=False, header=False)
    print(f"File saved with variance column at: {output_path}")
    return output_path

def plot_variance(df, output_filename, info, dim, nb):
    df.rename(columns={0: 'nq', df.columns[-2]: 'Avg', df.columns[-1]: 'Variance'}, inplace=True)

    if 'nq' not in df.columns or 'Variance' not in df.columns:
        print("Error: 'nq' or 'Variance' column not found in the CSV file.")
        print("Available columns:", df.columns)
    else:
        # Extract nq values and variance
        nq_values = df['nq']
        variances = df['Variance']

        plt.figure(figsize=(10, 6))
        plt.plot(nq_values, variances, marker='o', linestyle='-', color='r')

        plt.xlabel('Batch Size (nq)')
        plt.ylabel('Variance')
        plt.title(f'{info} Variance vs. Batch Size with dim={dim}, nb={nb}')
        plt.grid(True)
        max_variance = max(variances)
        plt.ylim(0, max_variance + 30)

        # Save the plot to a file
        plt.savefig(output_filename, format='pdf')
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 add_var.py <csv_filename> <output_dir>")
        exit()
    csv_filename = sys.argv[1]
    output_dir = sys.argv[2]
    pos = csv_filename.rfind("/")
    if pos == -1:
        plot_file_name = csv_filename
    else:
        plot_file_name = csv_filename[pos+1:]
    output_pathcsv = os.path.join(output_dir, plot_file_name)
    output_pathname = output_pathcsv.replace(".csv", "_variance_plot.pdf")

    # Extract information before '_dim'
    match_info = re.search(r'(.+)_dim_(\d+)_nb_(\d+)', csv_filename)
    if match_info:
        info = match_info.group(1)  # Info before '_dim'
        dim = match_info.group(2)
        nb = match_info.group(3)
    else:
        print("Error: Could not parse 'info', 'dim', and 'nb' from the file name.")
        info = "unknown"
        dim = "unknown"
        nb = "unknown"

    df = load_csv(csv_filename)
    df = compute_variance(df)  # Add variance column
    save_with_variance(df, csv_filename, output_dir)
    plot_variance(df, output_pathname, info, dim, nb)  # Plot variance against nq