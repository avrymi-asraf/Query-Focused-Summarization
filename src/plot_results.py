import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator


if __name__ == '__main__':

    # --- Configuration ---
    # Make sure your CSV file is named this or change the filename below
    filename = '../output/STH-01-run-all.csv'
    output_filename = 'accuracy_over_time.png'

    # --- 1. Load and Process the Data ---
    try:
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(filename)
        print(f"Successfully loaded '{filename}'. Found {len(df)} rows.")
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        print("Please make sure the script is in the same directory as your CSV file.")
        exit()

    # Group the data by the iteration number and calculate the mean for each column
    # This gives us the average accuracy at each step of the process
    average_accuracy_per_iteration = df.groupby('iteration_number')[['accuracy', 'acu_accuracy']].mean()

    # Reset the index so 'iteration_number' becomes a regular column for plotting
    average_accuracy_per_iteration = average_accuracy_per_iteration.reset_index()

    print("\nCalculated average accuracies per iteration:")
    print(average_accuracy_per_iteration.head())

    # --- 2. Create the Plot ---
    # Set a professional plot style
    sns.set_theme(style="whitegrid")

    # Create a figure and axes for the plot
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot the 'Overall Judge Accuracy'
    sns.lineplot(
        data=average_accuracy_per_iteration,
        x='iteration_number',
        y='accuracy',
        ax=ax,
        marker='o',  # Use circles for markers
        linestyle='-', # Use a solid line
        label='Overall Judge Accuracy'
    )

    # Plot the 'ACU Accuracy'
    sns.lineplot(
        data=average_accuracy_per_iteration,
        x='iteration_number',
        y='acu_accuracy',
        ax=ax,
        marker='s',  # Use squares for markers
        linestyle='--',# Use a dashed line
        label='ACU Accuracy'
    )

    # --- 3. Customize and Save the Plot ---
    # Set titles and labels
    ax.set_title('Average Accuracy Improvement per Iteration', fontsize=14, fontweight='bold')
    ax.set_xlabel('Iteration Number', fontsize=12)
    ax.set_ylabel('Average Accuracy', fontsize=12)

    # Set the y-axis to range from 0.0 to 1.0 for clarity
    ax.set_ylim(0, 1.05)

    # Ensure the x-axis has integer ticks (1, 2, 3, etc.)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(left=0.5) # Start the x-axis just before 1

    # Add a legend to identify the lines
    ax.legend()

    # Ensure everything fits without being cramped
    plt.tight_layout()

    # Save the plot to a file with high resolution for the paper
    plt.savefig(output_filename, dpi=300)

    print(f"\nPlot successfully saved as '{output_filename}'")

    # Optionally, display the plot on the screen
    plt.show()