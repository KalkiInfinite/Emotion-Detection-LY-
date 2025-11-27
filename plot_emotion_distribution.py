import matplotlib.pyplot as plt

# Data for the bar graph
emotions = ['happy', 'contempt', 'sad', 'fear', 'surprise', 'neutral', 'angry', 'disgust']
counts = ['21,796', '9,891', '16,062', '14,374', '14,389', '20,537', '14,766', '10,162']
colors = [
    (0, 255, 255),  # Yellow for happy
    (0, 100, 255),  # Orange for contempt
    (255, 0, 0),    # Blue for sad
    (255, 0, 255),  # Magenta for fear
    (255, 255, 0),  # Cyan for surprise
    (128, 128, 128),# Gray for neutral
    (0, 0, 255),    # Red for angry
    (0, 255, 0)     # Green for disgust
]

# Normalize colors to [0, 1] range for matplotlib
colors = [(r/255, g/255, b/255) for r, g, b in colors]

# Create the bar graph
plt.figure(figsize=(10, 6))
bars = plt.bar(emotions, [int(count.replace(',', '')) for count in counts], color=colors)

# Add exact numbers on top of each bar
for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500, count, ha='center', fontsize=10)

# Add labels and title
plt.xlabel('Emotions', fontsize=12)
plt.ylabel('Counts', fontsize=12)
plt.title('Emotion Distribution', fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Show the graph
plt.tight_layout()
plt.show()
