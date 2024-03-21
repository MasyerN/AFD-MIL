


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Your data goes here
data = {
    'Pathologist': ['P1', 'P2', 'P3', 'P4', 'P5', 'Combined Experience'],
    'MIL_Gaze_ACC': [0.861, 0.8218, 0.8157, 0.8187, 0.8595, 0.8112],
    'MIL_Gaze_Sen': [0.9103, 0.8897, 0.8846, 0.8821, 0.8974, 0.8821],
    'MIL_HPDA_ACC': [0.9245, 0.9048, 0.8489, 0.9139, 0.8746, 0.8218],
    'MIL_HPDA_Sen': [0.9179, 0.9077, 0.8487, 0.9077, 0.8615, 0.8307],
    'HPDA_ACC': [0.9522, 0.9497, 0.9491, 0.9484, 0.9491, 0.9305],
    'HPDA_Sen': [0.977, 0.9872, 0.9897, 0.9897, 0.9821, 0.9692]
}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# Function to create a radar chart with customizable scale and no axis labels
def create_radar_chart(df, title, min_val, max_val, labels, fig_size):
    # Number of variables we're plotting.
    num_vars = len(df.columns) - 1

    # Compute angle each bar is centered on:
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    # Set up the figure size
    fig, ax = plt.subplots(figsize=fig_size, subplot_kw=dict(polar=True))

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], labels, color='black', size=12)  # Adjust label color and size here

    # Set gridlines (but with no labels)
    ax.set_rlabel_position(0)
    ax.set_yticks(np.linspace(min_val, max_val, num=5))  # Define the grid lines but don't label them
    ax.set_yticklabels([])  # No labels for the grid lines
    plt.ylim(min_val, max_val)

    # Plot data + fill for each pathologist
    for idx, row in df.iterrows():
        data = row.drop('Pathologist').tolist()
        data += data[:1]
        ax.plot(angles, data, linewidth=1, linestyle='solid', label=row['Pathologist'])
        ax.fill(angles, data, alpha=0.25)

    # Add legend outside of the plot area
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.003), fancybox=True, shadow=True, ncol=3, fontsize=12)

    # Add title with adjusted color and size
    #plt.title(title, size=20, color='black', y=1.1)

    # Remove the frame of the chart
    ax.spines['polar'].set_visible(False)

    # Show the plot
    plt.show()
    plt.savefig('/home/omnisky/hdd_15T_sdd/rader.png', dpi=1000)

# Define the minimum and maximum values for your scale
min_val = 0.77  # Set the minimum value here
max_val = 0.99  # Set the maximum value here
labels = ['MIL Gaze ACC', 'MIL Gaze Sensitivity', 'MIL HPDA ACC', 'MIL HPDA Sensitivity', 'HPDA ACC', 'HPDA Sensitivity']  # Your labels
fig_size = (10, 6)  # You can adjust the figure size here (width, height)

# Create the radar chart with the given title and figure size
create_radar_chart(df, 'Pathologist Performance Characteristics', min_val, max_val, labels, fig_size)
