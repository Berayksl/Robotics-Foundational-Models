# save as visualize_houses.py
import prior
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import ai2thor.controller
from environment.stretch_controller import StretchController
import sys
import os
from utils.constants.stretch_initialization_utils import (
    STRETCH_ENV_ARGS,
)
from utils.constants.objaverse_data_dirs import OBJAVERSE_HOUSES_DIR

import math

def generate_circular_path(center_x, center_z, radius, num_points, y=0.9):
    path = []
    for i in range(num_points):
        theta = 2 * math.pi * i / num_points
        x = center_x + radius * math.cos(theta)
        z = center_z + radius * math.sin(theta)
        path.append({"x": x, "y": y, "z": z})
    return path


class HouseVisualizer:
    def __init__(self):
        """Initialize the house visualizer"""
        print("Loading houses...")
        self.houses = list(self.load_houses())
        self.controller = None
    
    # def load_houses(self):
    #     path = "/home/bera/Desktop/Codes/SPOC/spoc-robot-training/Evaluation/objaverse_houses/houses_2023_07_28/val.jsonl.gz"
    #     houses = []
    #     with gzip.open(path, "rt") as f:
    #         for line in f:
    #             line = line.strip()
    #             if not line:
    #                 continue
    #             houses.append(json.loads(line))
    #     return houses

    def load_houses(self):
         subset_to_load = "val"
         return prior.load_dataset(
            dataset="spoc-data",
            entity="spoc-robot",
            revision="local-objaverse-procthor-houses",
            path_to_splits=None,
            split_to_path={
                k: os.path.join(OBJAVERSE_HOUSES_DIR, f"{k}.jsonl.gz")
                for k in ["train", "val", "test"]
            },
            max_houses_per_split=int(1e9),
        )[subset_to_load]

                
    # def load_houses(self):
    #     """Load available houses"""
    #     houses = prior.load_dataset(
    #         dataset="spoc-data",
    #         entity="spoc-robot",
    #         revision="houses-test-val"
    #     )["val"]
    #     print(f"Loaded {len(houses)} houses")
    #     return houses
    
    def init_controller(self, house):
        """Initialize AI2-THOR controller with a house"""
        if self.controller is not None:
            self.controller.stop()

        controller_args = STRETCH_ENV_ARGS.copy()
        controller_args['renderInstanceSegmentation'] = False
        controller_args['server_timeout'] = 10
        controller_args['width'] = 800
        controller_args['height'] = 600

        self.controller = StretchController(
            scene=house,
            **controller_args
        )
        return self.controller
    
    def get_house_layout(self, house_index):
        """Get top-down view of a house"""
        #print('houses:',self.houses)
        house = self.houses[house_index]

        print('house:', house)

        #print(f"Loading house {house_index}: {house}")
        
        # Initialize controller with this houseq
        controller = self.init_controller(house)
        
        # Get top-down frame
        # Create a simple path through the house for visualization
        # event = controller.step("GetReachablePositions")
        # reachable_positions = event.metadata["actionReturn"]
        
        # if reachable_positions:
        #     # Create a path through some positions
        #     path = reachable_positions[:min(10, len(reachable_positions))]
        # else:
        #     # Default path if no reachable positions
        #path = [{"x": 0, "y": 0.9, "z": 5}, {"x": 1, "y": 0.9, "z": 5}, {"x": 2, "y": 0.9, "z": 6},{"x": 3, "y": 0.9, "z": 7}, {"x": 4, "y": 0.9, "z": 8}, {"x": 5, "y": 0.9, "z": 9}]
        
        center_x = 1.5
        center_z = 7.5
        radius = 0.5
        num_points = 20   # resolution of the circle

        path = generate_circular_path(center_x, center_z, radius, num_points)
        
        # Get top-down view
        top_down = controller.get_top_down_path_view(
            agent_path=path
        )
        return top_down, house
    
    def visualize_house(self, house_index):
        """Visualize a specific house"""
        print(f"\nVisualizing house {house_index}...")
        
        top_down, house = self.get_house_layout(house_index)
        # Display house info
        # rooms = house.get("rooms", [])
        # room_types = [r.get("roomType", "unknown") for r in rooms]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Show top-down view
        ax1.imshow(top_down)
        ax1.set_title(f"House {house_index} - Top Down View")
        ax1.axis('off')
        
        # Show house info
        ax2.axis('off')
        info_text = f"House Index: {house_index}\n"
        #info_text += f"Number of Rooms: {len(rooms)}\n"
        info_text += f"Room Types:\n"
        # for i, room_type in enumerate(room_types):
        #     info_text += f"  {i+1}. {room_type}\n"
        
        ax2.text(0.1, 0.9, info_text, transform=ax2.transAxes,
                fontsize=12, verticalalignment='top')
        
        plt.suptitle(f"House {house_index} Layout", fontsize=16)
        plt.tight_layout()
        return fig
    
    def browse_houses_interactive(self):
        """Interactive house browser"""
        current_index = 0
        
        print("\n" + "="*60)
        print("INTERACTIVE HOUSE BROWSER")
        print("="*60)
        print("Controls:")
        print("  n/→ : Next house")
        print("  p/← : Previous house")
        print("  s   : Select current house")
        print("  q   : Quit")
        print("  [number] : Jump to house index")
        print("="*60)
        
        plt.ion()  # Interactive mode
        fig = None
        
        while True:
            # Close previous figure if exists
            if fig is not None:
                plt.close(fig)
            
            # Visualize current house
            try:
                fig = self.visualize_house(current_index)
                plt.show()
                plt.pause(0.1)
            except Exception as e:
                print(f"Error visualizing house {current_index}: {e}")
                print("Try another house...")
            
            # Get user input
            command = input(f"\nCurrent: House {current_index} | Command: ").strip().lower()
            
            if command in ['n', '']:
                current_index = (current_index + 1) % len(self.houses)
            elif command == 'p':
                current_index = (current_index - 1) % len(self.houses)
            elif command == 's':
                print(f"\nSelected house {current_index}")
                if self.controller:
                    self.controller.stop()
                plt.close('all')
                return current_index, self.houses[current_index]
            elif command == 'q':
                print("Exiting...")
                if self.controller:
                    self.controller.stop()
                plt.close('all')
                return None, None
            elif command.isdigit():
                new_index = int(command)
                if 0 <= new_index < len(self.houses):
                    current_index = new_index
                else:
                    print(f"Invalid index. Must be 0-{len(self.houses)-1}")
    
    def quick_preview(self, num_houses=5):
        """Quick preview of multiple houses in one figure"""
        fig, axes = plt.subplots(1, num_houses, figsize=(20, 4))
        
        for i in range(num_houses):
            try:
                top_down, house = self.get_house_layout(i)
                
                if num_houses == 1:
                    ax = axes
                else:
                    ax = axes[i]
                
                ax.imshow(top_down)
                rooms = house.get("rooms", [])
                ax.set_title(f"House {i}\n{len(rooms)} rooms", fontsize=10)
                ax.axis('off')
            except Exception as e:
                print(f"Error with house {i}: {e}")
        
        plt.suptitle("House Preview", fontsize=14)
        plt.tight_layout()
        plt.show()
        
        return fig


def main():
    """Main function to run the house visualizer"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize and select houses")
    parser.add_argument("--mode", choices=["browse", "preview", "specific"], 
                       default="browse", help="Visualization mode")
    parser.add_argument("--house_index", type=int, default=0,
                       help="Specific house index to view")
    parser.add_argument("--num_preview", type=int, default=5,
                       help="Number of houses to preview")
    
    args = parser.parse_args()
    
    visualizer = HouseVisualizer()
    
    if args.mode == "browse":
        # Interactive browsing
        selected_index, selected_house = visualizer.browse_houses_interactive()
        if selected_house:
            print(f"\nFinal selection: House {selected_index}")
            # Save selection
            import json
            with open('selected_house.json', 'w') as f:
                json.dump({'house_index': selected_index}, f)
            print("Selection saved to selected_house.json")
    
    elif args.mode == "preview":
        # Quick preview
        visualizer.quick_preview(args.num_preview)
        plt.show(block=True)
    
    elif args.mode == "specific":
        # View specific house
        fig = visualizer.visualize_house(args.house_index)
        plt.show(block=True)
    
    # Cleanup
    if visualizer.controller:
        visualizer.controller.stop()


if __name__ == "__main__":
    main()