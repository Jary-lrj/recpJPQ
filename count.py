import numpy as np


def calculate_sparsity(file_path):
    user_item_pairs = set()
    users = set()
    items = set()

    # Read the file and parse user-item pairs
    with open(file_path, 'r') as file:
        for line in file:
            user_id, item_id = map(int, line.strip().split(' '))
            user_item_pairs.add((user_id, item_id))
            users.add(user_id)
            items.add(item_id)

    num_users = len(users)
    num_items = len(items)
    num_interactions = len(user_item_pairs)

    # Calculate possible interactions
    num_possible_interactions = num_users * num_items

    # Calculate sparsity
    sparsity = 1 - (num_interactions / num_possible_interactions)

    return sparsity, num_users, num_items, num_interactions, num_possible_interactions


if __name__ == "__main__":
    file_path = './data/movies_seg4.txt'  # Replace with your file path
    sparsity, num_users, num_items, num_interactions, num_possible_interactions = calculate_sparsity(
        file_path)

    print(f"Number of Users: {num_users}")
    print(f"Number of Items: {num_items}")
    print(f"Number of Interactions: {num_interactions}")
    print(f"Possible Interactions: {num_possible_interactions}")
    print(f"Sparsity: {sparsity:.4f}")
