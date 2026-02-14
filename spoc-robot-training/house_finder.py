#finds the room indices with the desired number of rooms from the objaverse dataset
import sys
import os
import prior

sys.path.append('/home/bera/Desktop/Codes/SPOC/spoc-robot-training/Evaluation')


def load_objaverse_houses():
    subset_to_load = "val"
    return prior.load_dataset(
        dataset="spoc-data",
        entity="spoc-robot",
        revision="local-objaverse-procthor-houses",
        path_to_splits=None,
        split_to_path={
            k: os.path.join(
                '/home/bera/Desktop/Codes/SPOC/spoc-robot-training/Evaluation/objaverse_houses',
                f"{k}.jsonl.gz"
            )
            for k in ["train", "val", "test"]
        },
        max_houses_per_split=int(1e9),
    )[subset_to_load]


def find_houses_with_n_rooms(houses, n_rooms):
    matching_indices = []

    for idx, house in enumerate(houses):
        if house is None:
            continue

        num_rooms = len(house.get("rooms", []))
        if num_rooms == n_rooms:
            matching_indices.append(idx)

    return matching_indices


if __name__ == "__main__":
    houses =list(load_objaverse_houses())
    n = int(input("Enter number of rooms: "))
    indices = find_houses_with_n_rooms(houses, n)

    print(f"Found {len(indices)} houses with {n} rooms.")
    print("House indices:", indices)
