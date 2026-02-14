# save as select_house.py
import prior
import argparse

def browse_houses():
    """Browse available houses and select one"""
    
    print("Loading available houses...")
    houses = prior.load_dataset(
        dataset="spoc-data",
        entity="spoc-robot",
        revision="houses-test-val"
    )["val"]
    
    print(f"\nAvailable houses: {len(houses)}")
    print("-" * 50)
    
    # Show first 10 houses
    for i in range(min(10, len(houses))):
        house = houses[i]
        num_rooms = len(house.get("rooms", []))
        room_types = [r.get("roomType", "unknown") for r in house.get("rooms", [])]
        print(f"House {i}: {num_rooms} rooms - {', '.join(room_types)}")
    
    # Let user select
    while True:
        try:
            choice = input("\nEnter house index (0-{}): ".format(len(houses)-1))
            house_idx = int(choice)
            if 0 <= house_idx < len(houses):
                return houses[house_idx], house_idx
            else:
                print("Invalid index")
        except ValueError:
            print("Please enter a number")

if __name__ == "__main__":
    selected_house, idx = browse_houses()
    print(f"\nSelected house {idx}")
    print(f"Rooms: {len(selected_house.get('rooms', []))}")
    
    # Save selection for use in evaluation
    import json
    with open('selected_house.json', 'w') as f:
        json.dump({'house_index': idx}, f)