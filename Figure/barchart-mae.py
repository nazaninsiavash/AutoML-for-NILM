import matplotlib.pyplot as plt
import numpy as np

# Define your data
models = [
    'DT', 'RF', 'GRU', 'LSTM', 'FCNN', 'DAE',
    'FHMM', 'CO', 'Window GRU', 'Seq2Point', 'Seq2Seq'
]
mae = [12.78, 11.6, 9.2, 8.63, 33.07, 9.7, 177.3, 224, 85.82, 8.31, 9.3]

# Create figure
plt.figure(figsize=(12, 7))

# Create bar chart
bars = plt.bar(models, mae, color=plt.cm.tab20(np.linspace(0, 1, len(models))))

# Add labels and title
plt.xlabel('Models', fontsize=14)
plt.ylabel('Mean Absolute Error (MAE)', fontsize=14)
plt.title('MAE of Different Models', fontsize=16)
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Annotate each bar with its MAE value
for bar, err in zip(bars, mae):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 5, f'{err:.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()
