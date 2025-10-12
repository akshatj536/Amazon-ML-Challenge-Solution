import pandas as pd

def count_max_bullet_points(file_path, chunksize=10000):
    """
    Counts the maximum number of bullet points in the 'catalog_content' column of a CSV file.

    Args:
        file_path (str): The path to the input CSV file.
        chunksize (int): The number of rows to process in each chunk.

    Returns:
        int: The maximum number of bullet points found.
    """
    max_bullets = 0
    try:
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            for content in chunk['catalog_content']:
                if isinstance(content, str):
                    bullet_count = content.count('Bullet Point')
                    if bullet_count > max_bullets:
                        max_bullets = bullet_count
    except FileNotFoundError:
        print(f"Error: Input file not found at {file_path}")
        return 0
    return max_bullets

if __name__ == '__main__':
    train_input_path = '/teamspace/studios/this_studio/dataset/train.csv'
    test_input_path = '/teamspace/studios/this_studio/dataset/test.csv'

    print("Counting max bullet points in train.csv...")
    max_bullets_train = count_max_bullet_points(train_input_path)
    print(f"Max bullet points in train.csv: {max_bullets_train}")

    print("Counting max bullet points in test.csv...")
    max_bullets_test = count_max_bullet_points(test_input_path)
    print(f"Max bullet points in test.csv: {max_bullets_test}")

    print(f"Overall max bullet points: {max(max_bullets_train, max_bullets_test)}")
