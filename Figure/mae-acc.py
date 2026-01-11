import matplotlib.pyplot as plt
import numpy as np

# Define your data
models = [
    'DT', 'RF', 'GRU', 'LSTM', 'FCNN', 'DAE',
    'FHMM', 'CO', 'Window GRU', 'Seq2Point', 'Seq2Seq'
]
accuracy = [99.12, 98.83, 97.98, 97.97, 98.94, 97.98, 94.9, 94.95, 80.9, 97.98, 97.96]
mae = [12.78, 11.6, 9.2, 8.63, 33.07, 9.7, 177.3, 224, 85.82, 8.31, 9.3]

# Create a color map
colors = plt.cm.tab20(np.linspace(0, 1, len(models)))

# Create scatter plot
plt.figure(figsize=(12, 8))

for i, model in enumerate(models):
    plt.scatter(mae[i], accuracy[i], color=colors[i], label=model, s=100)

# Label axes and title
plt.xlabel('Mean Absolute Error (MAE)', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.title('Accuracy vs MAE for Different Models', fontsize=16)
plt.grid(True)

# Show legend (outside plot)
plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

plt.tight_layout()
plt.show()
