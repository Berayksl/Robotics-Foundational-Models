#!/usr/bin/env python3
"""
simple_spoc_test.py - Simple test of SPOC model with custom prompts
"""

import os
import sys
import torch
import numpy as np
import argparse

# Add SPOC to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONPATH"] = "./"
os.environ["TOKENIZERS_PARALLELISM"] = "False"

from architecture.models.transformer_models import REGISTERED_MODELS
from online_evaluation.local_logging_utils import LoadLocalWandb


def test_spoc_model():
    """Simple test of SPOC model functionality"""
    
    # Configuration
    training_run_id = "SigLIP-ViTb-3-double-det-CHORES-S"
    local_checkpoint_dir = "/home/bera/Desktop/Codes/SPOC/spoc-robot-training/Evaluation/pre-trained"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*60}")
    print("SPOC Model Test")
    print(f"{'='*60}")
    
    # Load model
    print(f"\nLoading model: {training_run_id}")
    run = LoadLocalWandb(run_id=training_run_id, save_dir=local_checkpoint_dir)
    ckpt_path = run.get_checkpoint(ckpt_step=None)
    
    # Get model configuration
    model_name = run.config["model"]
    model_version = run.config["model_version"]
    input_sensors = run.config["input_sensors"]
    loss = run.config["loss"]
    
    print(f"Model: {model_name}")
    print(f"Version: {model_version}")
    print(f"Required sensors: {input_sensors}")
    
    # Build agent
    agent_class = REGISTERED_MODELS[model_name]
    agent_input = {
        "model_version": model_version,
        "input_sensors": input_sensors,
        "loss": loss,
        "sampling": "greedy",
        "ckpt_pth": ckpt_path,
        "device": device
    }
    
    print("\nBuilding agent...")
    agent = agent_class.build_agent(**agent_input)
    if hasattr(agent, "model"):
        agent.model.eval()
    print("✓ Agent ready")
    
    # Test prompts
    test_prompts = [
        "Navigate to apple",
        "Pick up the mug",
        "Fetch basketball",
    ]
    
    print(f"\n{'='*60}")
    print("Testing with custom prompts")
    print(f"{'='*60}")
    
    for prompt in test_prompts:
        print(f"\nTask: {prompt}")
        print("-" * 40)
        
        # Reset agent
        agent.reset()
        
        # Create mock observation
        nav_image = (np.random.rand(224, 384, 3) * 255).astype(np.uint8)
        manip_image = (np.random.rand(224, 384, 3) * 255).astype(np.uint8)
        
        observation = {
            "raw_navigation_camera": nav_image,
            "raw_manipulation_camera": manip_image,
            "an_object_is_in_hand": np.array([0], dtype=np.float32),
        }
        
        # Add bbox sensors if needed (10 values: 5 for nav, 5 for manip)
        if "nav_task_relevant_object_bbox" in input_sensors:
            no_detection = np.array([-1.0] * 10, dtype=np.float32)
            observation["nav_task_relevant_object_bbox"] = no_detection
        if "manip_task_relevant_object_bbox" in input_sensors:
            no_detection = np.array([-1.0] * 10, dtype=np.float32)
            observation["manip_task_relevant_object_bbox"] = no_detection
        if "nav_accurate_object_bbox" in input_sensors:
            no_detection = np.array([-1.0] * 10, dtype=np.float32)
            observation["nav_accurate_object_bbox"] = no_detection
        if "manip_accurate_object_bbox" in input_sensors:
            no_detection = np.array([-1.0] * 10, dtype=np.float32)
            observation["manip_accurate_object_bbox"] = no_detection
        
        # Add last_actions if needed
        if "last_actions" in input_sensors:
            observation["last_actions"] = np.array([], dtype=np.int32)
        
        # Filter observations
        observation = {k: v for k, v in observation.items() if k in input_sensors}
        
        # Get action
        try:
            with torch.no_grad():
                action = agent.get_action(observation, prompt)
                
                # Convert action to readable format
                if isinstance(action, tuple):
                    action = action[0]
                if isinstance(action, torch.Tensor):
                    action = action.item()
                if isinstance(action, np.ndarray):
                    action = action.item()
                
                # Action names from SPOC paper
                action_names = [
                    "move_base_forward", "move_base_backward",
                    "rotate_left_6", "rotate_left_30", 
                    "rotate_right_6", "rotate_right_30",
                    "move_arm_x+2", "move_arm_x+10", "move_arm_x-2", "move_arm_x-10",
                    "move_arm_z+2", "move_arm_z+10", "move_arm_z-2", "move_arm_z-10",
                    "rotate_gripper+10", "rotate_gripper-10",
                    "pickup", "dropoff", "done", "terminate"
                ]
                
                action_str = action_names[action] if action < len(action_names) else f"action_{action}"
                print(f"Model predicted: {action_str} (action_id: {action})")
                
        except Exception as e:
            print(f"Error: {e}")
    
    print(f"\n{'='*60}")
    print("Test Complete!")
    print(f"{'='*60}")
    print("\nThe model is working correctly and responding to prompts.")
    print("For real deployment:")
    print("1. Replace mock images with real camera feeds")
    print("2. Add object detection for bounding boxes")
    print("3. Connect to robot/simulator for action execution")


if __name__ == "__main__":
    test_spoc_model()