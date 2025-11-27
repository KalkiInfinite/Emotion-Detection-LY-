import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Confusion matrix data
confusion_matrix = [
    [797, 6, 32, 45, 51, 186, 167, 65],
    [3, 58, 6, 0, 21, 48, 8, 3],
    [45, 8, 99, 0, 11, 13, 28, 7],
    [143, 0, 11, 279, 29, 143, 311, 222],
    [46, 13, 20, 20, 2666, 120, 68, 104],
    [67, 22, 10, 25, 122, 1961, 369, 96],
    [136, 2, 14, 60, 48, 372, 1068, 59],
    [39, 5, 12, 33, 42, 51, 29, 1210]
]

# Emotion labels
emotion_labels = [
    "angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"
]

# Normalize the confusion matrix
confusion_matrix_normalized = np.array(confusion_matrix) / np.sum(confusion_matrix, axis=1, keepdims=True)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=emotion_labels, yticklabels=emotion_labels)
plt.xlabel("Predicted Labels", fontsize=12)
plt.ylabel("True Labels", fontsize=12)
plt.title("Confusion Matrix", fontsize=14)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
