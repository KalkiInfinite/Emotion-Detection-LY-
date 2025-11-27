import os

def count_images_in_emotion_folders(base_path):
    """
    Count the number of images for each emotion across train, val, and test directories.

    Args:
        base_path (str): The base directory containing train, val, and test folders.

    Returns:
        dict: A dictionary with emotions as keys and their total image counts as values.
    """
    subsets = ['train', 'val', 'test']
    emotion_counts = {}

    for subset in subsets:
        subset_path = os.path.join(base_path, subset)
        if not os.path.exists(subset_path):
            print(f"Warning: {subset_path} does not exist.")
            continue

        for emotion in os.listdir(subset_path):
            emotion_path = os.path.join(subset_path, emotion)
            if os.path.isdir(emotion_path):
                image_count = len([f for f in os.listdir(emotion_path) if os.path.isfile(os.path.join(emotion_path, f))])
                if emotion not in emotion_counts:
                    emotion_counts[emotion] = 0
                emotion_counts[emotion] += image_count

    return emotion_counts

if __name__ == "__main__":
    base_path = "data/unified_dataset"  # Update this path if needed
    emotion_counts = count_images_in_emotion_folders(base_path)

    print("Emotion counts:")
    for emotion, count in emotion_counts.items():
        print(f"{emotion}: {count}")
