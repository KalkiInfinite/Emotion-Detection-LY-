import os
import matplotlib.pyplot as plt
from collections import Counter

def visualize_emotion_distribution(dataset_path):
    """
    Visualize the distribution of emotions in the dataset as a bar graph.

    Args:
        dataset_path (str): Path to the dataset directory. The dataset should be organized
                           with subdirectories for each emotion category.
    """
    # Check if the dataset path exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' does not exist.")
        return

    # Get the list of emotion categories (subdirectories)
    emotion_categories = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

    # Count the number of images in each category
    emotion_counts = {}
    for category in emotion_categories:
        category_path = os.path.join(dataset_path, category)
        num_images = len([f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))])
        emotion_counts[category] = num_images

    # Sort the categories alphabetically for consistent visualization
    emotion_counts = dict(sorted(emotion_counts.items()))

    # Plot the bar graph
    plt.figure(figsize=(10, 6))
    plt.bar(emotion_counts.keys(), emotion_counts.values(), color='skyblue')
    plt.xlabel('Emotion Categories', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.title('Emotion Distribution in Dataset', fontsize=14)
    plt.xticks(rotation=45, fontsize=10)
    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(dataset_path, 'emotion_distribution.png')
    plt.savefig(output_path)
    print(f"Bar graph saved as '{output_path}'")

    # Show the plot
    plt.show()

if __name__ == "__main__":
    dataset_path = input("Enter the path to your dataset: ").strip()
    visualize_emotion_distribution(dataset_path)
