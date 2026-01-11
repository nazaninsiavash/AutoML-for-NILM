import matplotlib.pyplot as plt
import numpy as np

# Define your data
models = [
    'DT', 'RF', 'GRU', 'LSTM', 'FCNN', 'DAE',
    'FHMM', 'CO', 'Window GRU', 'Seq2Point', 'Seq2Seq'
]
accuracy = [99.12, 98.83, 97.98, 97.97, 98.94, 97.98, 94.9, 94.95, 80.9, 97.98, 97.96]

# Create figure
plt.figure(figsize=(12, 7))

# Create bar chart
bars = plt.bar(models, accuracy, color=plt.cm.tab20(np.linspace(0, 1, len(models))))

# Add labels and title
plt.xlabel('Models', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.title('Accuracy of Different Models', fontsize=16)
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.ylim(0, 105)  # Set y-axis limit slightly above 100
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Annotate each bar with its accuracy value
for bar, acc in zip(bars, accuracy):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{acc:.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()
