##Hard filtering using future robustness degree and Q values (2/11/2026)

import multiprocessing as mp
import os
import platform
import sys
import traceback
from itertools import chain
from queue import Empty as EmptyQueueError
from typing import Literal, Optional, Dict, Any, cast, Sequence, List
import math

import ai2thor.platform
import numpy as np
import torch
from matplotlib import pyplot as plt
import cv2

from architecture.agent import AbstractAgent
from environment.manipulation_sensors import TargetObjectWasPickedUp
from environment.navigation_sensors import (
    BestBboxSensorOnlineEval,
    CurrentAgentRoom, 
    NumPixelsVisible,
    SlowAccurateObjectBBoxSensor,
    TaskRelevantObjectBBoxSensorDeticOnlineEvalDetic,
    TaskRelevantObjectBBoxSensorDummy,
    TaskRelevantObjectBBoxSensorOnlineEval,
)
from environment.stretch_controller import StretchController
from online_evaluation.max_episode_configs import MAX_EPISODE_LEN_PER_TASK
from online_evaluation.online_evaluation_types_and_utils import (
    calc_trajectory_room_visitation,
)
from tasks import AbstractSPOCTask
from tasks.object_nav_task import ObjectNavTask 
from tasks.multi_task_eval_sampler import MultiTaskSampler
from tasks.task_specs import TaskSpecDatasetList, TaskSpecQueue
from utils.constants.stretch_initialization_utils import (
    STRETCH_ENV_ARGS,
)
from utils.data_generation_utils.mp4_utils import save_frames_to_mp4
from utils.task_datagen_utils import (
    get_core_task_args,
    add_extra_sensors_to_task_args,
)
from utils.type_utils import THORActions
from utils.visualization_utils import add_bbox_sensor_to_image, get_top_down_frame, VideoLogging

import argparse
from allenact.utils.misc_utils import str2bool
import wandb
import datetime
import time
from online_evaluation.local_logging_utils import LoadLocalWandb, LocalWandb
from architecture.models.transformer_models import REGISTERED_MODELS
import prior
from environment.stretch_controller import StretchController
from environment.unicycle_controller import unicycle_step

from robustness_calculator import calculate_robustness

from RL.src.networks import QNetwork
from RL.src.main import select_model_file, normalize_state
from RL.src.simulator import Continuous2DEnv
from RL.src.dynamics import DiscreteUnicycleDynamics

def parse_args():
    parser = argparse.ArgumentParser(description="Online evaluation")
    parser.add_argument("--training_run_id", default='SigLIP-ViTb-3-double-det-CHORES-L' ,type=str)
    parser.add_argument("--ckptStep", default=None, type=int)
    parser.add_argument("--max_eps_len", default=-1, type=int)
    parser.add_argument("--eval_set_size", default=200, type=int)
    parser.add_argument("--sampling", default="sample")
    parser.add_argument("--gpu_devices", nargs="+", default=[0, 1], type=int)
    parser.add_argument("--num_workers", type=int, default = 1)
    parser.add_argument("--shuffle", action="store_true", default=False)
    parser.add_argument("--test_augmentation", action="store_true", default=False)
    parser.add_argument("--skip_done", action="store_true", default=False)
    parser.add_argument("--eval_subset", default="minival", help="options: val, minival, train")
    parser.add_argument("--dataset_type", default="")
    parser.add_argument("--task_type", default="")
    parser.add_argument("--det_type", default="gt", help="gt or detic", choices=["gt", "detic"])
    parser.add_argument("--house_set", default="procthor", help="procthor or objaverse")
    parser.add_argument("--dataset_path", default="/data/datasets")
    parser.add_argument("--output_basedir", default="tmp_log")
    parser.add_argument("--local_checkpoint_dir", default="/home/bera/Desktop/Codes/SPOC/spoc-robot-training/Evaluation/pre-trained")
    parser.add_argument("--extra_tag", default="")
    parser.add_argument("--benchmark_revision", default="chores-small")
    parser.add_argument("--wandb_logging", default=False, type=str2bool)
    parser.add_argument("--wandb_project_name", default="", type=str)
    parser.add_argument("--wandb_entity_name", default="", type=str)
    parser.add_argument(
        "--input_sensors",
        nargs="+",
        default=["raw_navigation_camera", "raw_manipulation_camera"],
    )
    parser.add_argument("--model_version_override", default="auto")
    parser.add_argument("--total_num_videos", type=int, default=8200)

    args = parser.parse_args()

    if len(args.gpu_devices) == 1 and args.gpu_devices[0] == -1:
        args.gpu_devices = None
    elif len(args.gpu_devices) == 0:
        # Get all the available GPUS
        args.gpu_devices = [i for i in range(torch.cuda.device_count())]

    if args.wandb_logging:
        assert args.wandb_project_name != ""
        assert args.wandb_entity_name != ""

    return args


def start_worker(worker, agent_class, agent_input, device, tasks_queue, results_queue):
    agent = agent_class.build_agent(**agent_input, device=device)
    if hasattr(agent, "model"):
        agent.model.eval()
    # add actor-critic model version for on-policy RL agents
    elif hasattr(agent, "actor_critic"):
        agent.actor_critic.eval()
    else:
        raise NotImplementedError
    try:
        # Keep working as long as there are tasks left to process
        worker.distribute_evaluate(agent, tasks_queue, results_queue)
    finally:
        # Notify the logger that there's nothing else to read from this worker
        try:
            results_queue.put(None)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            print(
                f"WARNING: Failed to put termination signal for worker {agent_input['worker_id']}"
            )
        # Regardless of whether there was an uncaught exception or the process finished, attempt to stop the controller.
        worker.stop()


class OnlineEvaluatorWorker:
    def __init__(
        self,
        gpu_device: int,
        houses: List[Dict[str, Any]],
        max_eps_len: int,
        input_sensors: Sequence[str],
        skip_done: bool,
        logging_sensor: "VideoLogging",
        outdir: str,
        worker_id: int,
        det_type: str,
    ):
        self.controller = None
        self.gpu_device = gpu_device
        self.houses = houses
        self.pre_defined_max_steps = max_eps_len
        self.input_sensors = input_sensors
        self.skip_done = skip_done
        self.logging_sensor: "VideoLogging" = logging_sensor
        self.outdir = outdir
        self.worker_id = worker_id
        self.det_type = det_type
        self._cached_sensors = None

        self._task_sampler: Optional[MultiTaskSampler] = None

    def get_house(self, sample):
        house_idx = int(sample["house_id"])
        print(self.houses[house_idx])
        return self.houses[house_idx], house_idx

    def get_agent_starting_position(self, sample):
        x, y, z = sample["observations"]["initial_agent_location"][:3]
        # TODO: change to an assert when pickup benchmark reprocessed
        y = 0.9009921550750732  # Brute force correction for old pickup task samples
        return dict(x=x, y=y, z=z)

    def get_agent_starting_rotation(self, sample):
        x, y, z = sample["observations"]["initial_agent_location"][3:]
        return dict(x=x, y=y, z=z)

    def get_extra_sensors(self):
        if self._cached_sensors is not None:
            return self._cached_sensors

        if self.det_type == "detic":
            nav_box_fast = TaskRelevantObjectBBoxSensorDeticOnlineEvalDetic(
                which_camera="nav", uuid="nav_task_relevant_object_bbox", gpu_device=self.gpu_device
            )
            nav_box_accurate = TaskRelevantObjectBBoxSensorDummy(
                which_camera="nav",
                uuid="nav_accurate_object_bbox",
            )
            manip_box_fast = TaskRelevantObjectBBoxSensorDeticOnlineEvalDetic(
                which_camera="manip",
                uuid="manip_task_relevant_object_bbox",
                gpu_device=self.gpu_device,
            )
            manip_box_accurate = TaskRelevantObjectBBoxSensorDummy(
                which_camera="manip",
                uuid="manip_accurate_object_bbox",
            )

        elif self.det_type == "gt":
            nav_box_fast = TaskRelevantObjectBBoxSensorOnlineEval(
                which_camera="nav", uuid="nav_task_relevant_object_bbox"
            )
            manip_box_fast = TaskRelevantObjectBBoxSensorOnlineEval(
                which_camera="manip", uuid="manip_task_relevant_object_bbox"
            )
            nav_box_accurate = TaskRelevantObjectBBoxSensorOnlineEval(
                which_camera="nav",
                uuid="nav_accurate_object_bbox",
                original_sensor_to_use=SlowAccurateObjectBBoxSensor,
            )
            manip_box_accurate = TaskRelevantObjectBBoxSensorOnlineEval(
                which_camera="manip",
                uuid="manip_accurate_object_bbox",
                original_sensor_to_use=SlowAccurateObjectBBoxSensor,
            )

        else:
            raise NotImplementedError(f"Unknown detection type {self.det_type}")

        best_bbox_nav = BestBboxSensorOnlineEval(
            which_camera="nav",
            uuid="nav_best_bbox",
            sensors_to_use=[nav_box_fast, nav_box_accurate],
        )
        best_bbox_manip = BestBboxSensorOnlineEval(
            which_camera="manip",
            uuid="manip_best_bbox",
            sensors_to_use=[manip_box_fast, manip_box_accurate],
        )
        extra_sensors = [
            CurrentAgentRoom(),
            NumPixelsVisible(which_camera="manip"),
            NumPixelsVisible(which_camera="nav"),
            #  Old setting
            nav_box_fast,
            manip_box_fast,
            #  New Setting
            nav_box_accurate,
            manip_box_accurate,
            # For metrics
            TargetObjectWasPickedUp(),
            best_bbox_nav,
            best_bbox_manip,
        ]

        self._cached_sensors = extra_sensors
        return extra_sensors

    def stop(self):
        try:
            if self._task_sampler is not None:
                self._task_sampler.close()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            print(f"WARNING: worker {self.worker_id} failed to stop with non-None task_sampler")
        finally:
            self._task_sampler = None

    @property
    def task_sampler(self) -> MultiTaskSampler:
        if self._task_sampler is None:
            task_args = get_core_task_args(max_steps=self.pre_defined_max_steps)

            add_extra_sensors_to_task_args(task_args, self.get_extra_sensors())

            self._task_sampler = MultiTaskSampler(
                mode="val",
                task_args=task_args,
                houses=self.houses,
                house_inds=list(range(len(self.houses))),
                controller_args={
                    **STRETCH_ENV_ARGS,
                    "platform": (
                        ai2thor.platform.OSXIntel64
                        if sys.platform.lower() == "darwin"
                        else ai2thor.platform.CloudRendering
                    ),
                },
                controller_type=StretchController,
                task_spec_sampler=TaskSpecDatasetList(
                    []
                ),  # Will be overwritten in distribute_evaluate
                visualize=False,
                prob_randomize_materials=0,
                device=self.gpu_device if self.gpu_device == "cpu" or self.gpu_device > 0 else None,
            )
        return self._task_sampler

    def evaluate_on_task(self, task: AbstractSPOCTask, agent: AbstractAgent, worker_id: int):
        global target_reached

        goal = task.task_info["natural_language_spec"]

        #print(task.task_info['house'])
        #print(task.task_info)

        object_type = task.task_info["synsets"][0]
        object_ids = task.task_info["synset_to_object_ids"][object_type]

        # print('object type:', object_type)
        # print('object ids:', object_ids)

        # task_path points out the episode's origin (i.e., which task, episode id, streaming id)
        task_path = "/".join(task.task_info["eval_info"]["task_path"].split("/")[-4:])

        all_frames = []
        all_video_frames = []
        agent.reset()
        action_list = agent.get_action_list()

        all_actions = []

        additional_metrics = {}

        target_reached = False
        main_task_done = False

        normalized_q_list = []
        normalized_logit_list = []
        regular_Q_vals = []
        regular_logits = []

        init_t = 30
        eps_idx = init_t
        num_of_available_actions_per_step = []

        with torch.no_grad():
            while eps_idx < task.max_steps:
                print("time step:", eps_idx)
                observations = task.get_observations()

                assert all(
                    input_sensor in observations
                    for input_sensor in self.input_sensors
                    if input_sensor != "last_actions"
                ), (
                    f"Observations do not contain all input sensors."
                    f" Observations: {observations.keys()}."
                    f" Input sensors: {self.input_sensors}"
                )

                observations = {k: v for k, v in observations.items() if k in self.input_sensors}


                curr_frame = np.concatenate(
                    [task.controller.navigation_camera, task.controller.manipulation_camera], axis=1
                )

                display_realtime = True
                # REAL-TIME DISPLAY
                if display_realtime:
                    # Add text overlay with task info
                    display_scale = 2

                    height, width = curr_frame.shape[:2]
                    new_width = int(width * display_scale)
                    new_height = int(height * display_scale)
                    
                    # Use INTER_LINEAR for smoother scaling
                    display_frame = cv2.resize(curr_frame, (new_width, new_height), 
                                              interpolation=cv2.INTER_LINEAR)
                    
                    # Add text overlays with larger font
                    font_scale = 1.0 * display_scale
                    thickness = max(2, int(2 * display_scale))
                    #display_frame = curr_frame.copy()
                    cv2.putText(display_frame, f"Task: {goal}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Step: {eps_idx + 1}/{task.max_steps}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Get and display action
                    action, logits = agent.get_action(observations, goal)
                    original_probs = torch.softmax(torch.tensor(logits), -1).detach().numpy()
                    #print("Original action:", action)
                    #print('action probs:',probs)
                    #print(f"probs: {probs}")
                    cv2.putText(display_frame, f"Action: {action}", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Show frame 
                    cv2.imshow('SPOC Evaluation - Press Q to quit', display_frame)
                    
                    # Wait for key press (1ms delay to allow display update)
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        print("User requested quit")
                        cv2.destroyAllWindows()
                        break
                else:
                    action, logits = agent.get_action(observations, goal)

                all_frames.append(curr_frame)
                
                if self.skip_done and action in ["end", "done"] and not main_task_done:
                    action = "sub_done"
                    main_task_done = True #whether the main task satisfied
                    main_task_done_step = eps_idx

                if action == "end" and not main_task_done:
                    main_task_done = True
                    main_task_done_step = eps_idx

                full_pose = task.controller.get_current_agent_full_pose() #(x, y, z) *y is height
                current_state = (full_pose['position']['x'], full_pose['position']['z'], full_pose['rotation']['y']) #(x, y, theta in degrees)

                if np.linalg.norm(current_state[:2] - np.array(goals[0]['center'])) <= goals[0]['radius'] and not target_reached:
                    target_reached = True
                    print("STL task satisifed at step", eps_idx)

                print("current state:", current_state)

                if action != "sub_done" and not target_reached:
                    regular_actions = ['m', 'b', 'l', 'r', 'ls', 'rs']
                    regular_action_logits = np.array([logits[action_list.index(a)] for a in regular_actions])

                    if not main_task_done:

                        for a in regular_actions:
                            future_trajectory = forward_propagate(env_2d, current_state, eps_idx, a, STL_horizon, Q_net)
                            future_robustness = calculate_robustness(future_trajectory, env_2d.goals[0]['center'], env_2d.goals[0]['radius'])
                            if future_robustness < 0:
                                print(f"Action {a} leads to STL violation with robustness {future_robustness}. Setting its logit to -inf.")
                                regular_action_logits[regular_actions.index(a)] = -float('inf')

                            else:
                                print("future trajectory for action", a, ":", future_trajectory)
                            # print(f"Future trajectory under action {a}:", future_trajectory)
                            # print('Trajectory length:', len(future_trajectory))

                        logits_tensor = torch.tensor(regular_action_logits, dtype=torch.float32, device=device)                
                        probs = torch.softmax(logits_tensor, -1)
                        action_idx = torch.distributions.categorical.Categorical(logits=logits_tensor).sample() #sample the action on the modified logits
                        action = regular_actions[action_idx]

                        #find the number of available actions (i.e., those that do not lead to STL violation)
                        available_actions = sum(regular_action_logits > -float('inf'))
                        num_of_available_actions_per_step.append(available_actions)


                    else:
                        #Take argmax(Q) after main task is done
                        state_norm = normalize_state(env_2d, current_state, device = device)
                        t_norm = float(eps_idx) / float(max(STL_horizon-1, 1))
                        t_tensor = torch.tensor([[t_norm]], dtype=torch.float32, device=device)  # (1,1)
                        x = torch.cat([state_norm, t_tensor], dim=1) 
                        Q_values = Q_net(x)

                        a_idx = Q_values.argmax(dim=1).item()
                        action = regular_actions[a_idx]
                        print(f"Main task done. Taking action with highest Q value: {action} with Q value {Q_values[0, a_idx].item()}")

                # print("Predicted next state:", unicycle_step(current_state, action)[0])
                print("Taken action:", action)
                all_actions.append(action)
                task.step_with_action_str(action)

                if "nav_best_bbox" in observations:
                    add_bbox_sensor_to_image(
                        curr_frame=curr_frame,
                        task_observations=observations,
                        det_sensor_key="nav_best_bbox",
                        which_image="nav",
                    )
                elif "nav_task_relevant_object_bbox" in observations:
                    add_bbox_sensor_to_image(
                        curr_frame=curr_frame,
                        task_observations=observations,
                        det_sensor_key="nav_task_relevant_object_bbox",
                        which_image="nav",
                    )
                if "manip_best_bbox" in observations:
                    add_bbox_sensor_to_image(
                        curr_frame=curr_frame,
                        task_observations=observations,
                        det_sensor_key="manip_best_bbox",
                        which_image="manip",
                    )
                elif "manip_task_relevant_object_bbox" in observations:
                    add_bbox_sensor_to_image(
                        curr_frame=curr_frame,
                        task_observations=observations,
                        det_sensor_key="manip_task_relevant_object_bbox",
                        which_image="manip",
                    )

                video_frame = self.logging_sensor.get_video_frame(
                    agent_frame=curr_frame,
                    frame_number=eps_idx,
                    action_names=action_list,
                    action_dist=original_probs.tolist(),
                    ep_length=task.max_steps,
                    last_action_success=task.last_action_success,
                    taken_action=action,
                    task_desc=goal,
                )

                all_video_frames.append(video_frame)
                
                # if task.is_done():
                #     print(f'Task is done at step {eps_idx}, breaking out of the loop.')
                #     break
                
                eps_idx += 1

                if main_task_done and target_reached:
                    break
        

        #plot Q-values and logits
        plt.figure(figsize=(12, 6))
        for a in regular_actions:
            a_index = regular_actions.index(a)
            plt.subplot(4, 1, 1)
            plt.plot(range(len(normalized_q_list)), [q[a_index] for q in normalized_q_list], label=f'Action {a}')
            plt.title('Normalized Q-values')
            plt.xlabel('Time step')
            plt.ylabel('Normalized Q-value')
            plt.grid(True)

            plt.subplot(4, 1, 2)
            plt.plot(range(len(normalized_logit_list)), [logit[a_index] for logit in normalized_logit_list], label=f'Action {a}')
            plt.title('Normalized Logits')
            plt.xlabel('Time step')
            plt.ylabel('Normalized Logit')
            plt.grid(True)
            
            plt.subplot(4, 1, 3)
            plt.plot(range(len(regular_Q_vals)), [q[a_index] for q in regular_Q_vals], label=f'Q {a}')
            plt.title('Q-values')
            plt.xlabel('Time step')
            plt.ylabel('Q-value')
            plt.grid(True)

            plt.subplot(4, 1, 4)
            plt.plot(range(len(regular_logits)), [logit[a_index] for logit in regular_logits], label=f'Logit {a}')
            plt.title('Logits')
            plt.xlabel('Time step')
            plt.ylabel('Logit')
            plt.grid(True)

        plt.legend()
        plt.tight_layout()
        plt.show()

        # #plot the normalized Q-value gaps
        # plt.figure(figsize=(12, 6))
        # normalized_spread = [q.max() - q.min() for q in normalized_q_list]
        # plt.plot(range(len(normalized_spread)), normalized_spread)
        # plt.title('Normalized Q-value gap (max - min) over time')
        # plt.xlabel('Time step')
        # plt.ylabel('Normalized Q-value gap')
        # plt.grid(True)
        # plt.show()

        # #plot the Q value gaps
        # plt.figure(figsize=(12, 6))
        # spread = [q.max() - q.min() for q in regular_Q_vals]
        # plt.plot(range(len(spread)), spread)
        # plt.title('Q-value gap (max - min) over time')
        # plt.xlabel('Time step')
        # plt.ylabel('Q-value gap')
        # plt.grid(True)
        # plt.show()

        # #plot the logit gaps
        # plt.figure(figsize=(12, 6))
        # spread = [logit.max() - logit.min() for logit in regular_logits]
        # plt.plot(range(len(spread)), spread)
        # plt.title('Logit gap (max - min) over time')
        # plt.xlabel('Time step')
        # plt.ylabel('Logit gap')
        # plt.grid(True)
        # plt.show()

        # #plot the normalized logit gaps
        # plt.figure(figsize=(12, 6))
        # normalized_spread = [logit.max() - logit.min() for logit in normalized_logit_list]
        # plt.plot(range(len(normalized_spread)), normalized_spread)
        # plt.title('Normalized Logit gap (max - min) over time')
        # plt.xlabel('Time step')
        # plt.ylabel('Normalized Logit gap')
        # plt.grid(True)
        # plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(range(init_t, len(num_of_available_actions_per_step)+init_t), num_of_available_actions_per_step)
        plt.title('Number of available actions (not leading to STL violation) per step')
        plt.xlabel('Time step')
        plt.ylabel('Number of available actions')
        plt.grid(True)
        plt.show()

        if display_realtime:
            cv2.destroyAllWindows()
            
        success = task.is_successful()

        if main_task_done:
            print(f'Main task satisfied at step {main_task_done_step}, breaking out of the loop.')

        print("Main task success:", main_task_done)
        print('STL task success:', target_reached)

        target_ids = None
        if "synset_to_object_ids" in task.task_info:
            target_ids = list(
                chain.from_iterable(task.task_info.get("synset_to_object_ids", None).values())
            )

        top_down_frame = get_top_down_frame(
            task.controller, task.task_info["followed_path"], target_ids
        )
        top_down_frame = np.ascontiguousarray(top_down_frame)

        metrics = self.calculate_metrics(
            task,
            all_actions,
            success,
            additional_metrics,
        )

        return dict(
            goal=goal,
            all_frames=all_frames,
            all_video_frames=all_video_frames,
            top_down_frame=top_down_frame,
            metrics=metrics,
            task_path=task_path,
        )

    def get_num_pixels_visible(self, which_camera: Literal["nav", "manip"], task):
        observations = task.get_observation_history()
        num_frames_visible = [obs[f"num_pixels_visible_{which_camera}"] for obs in observations]
        max_num_frame_obj_visible = max(num_frames_visible).item()
        return max_num_frame_obj_visible

    def has_agent_been_in_obj_room(self, task):
        observations = task.get_observation_history()

        object_type = task.task_info["synsets"][0]
        object_ids = task.task_info["synset_to_object_ids"][object_type]
        target_object_rooms = [
            task.controller.get_objects_room_id_and_type(obj_id)[0] for obj_id in object_ids
        ]
        target_object_rooms = [int(x.replace("room|", "")) for x in target_object_rooms]
        agents_visited_rooms = [obs["current_agent_room"].item() for obs in observations]
        visited_the_objects_room = [x for x in target_object_rooms if x in agents_visited_rooms]
        visited_objects_room = len(visited_the_objects_room) > 0
        return visited_objects_room

    def get_extra_per_obj_metrics(self, task, metrics):
        try:
            object_type = task.task_info["synsets"][0]

            if metrics["success"] < 0.1:
                metrics[f"extra/{object_type}/when_failed_visited_obj_room"] = (
                    self.has_agent_been_in_obj_room(task)
                )

                metrics[f"extra/{object_type}/when_failed_max_visible_pixels_navigation"] = (
                    self.get_num_pixels_visible("nav", task)
                )

                metrics[f"extra/{object_type}/when_failed_max_visible_pixels_manipulation"] = (
                    self.get_num_pixels_visible("manip", task)
                )

            metrics[f"extra/{object_type}/success"] = metrics[
                "success"
            ]  # This should be different for different tasks
            metrics[f"extra/{object_type}/eps_len"] = metrics[
                "eps_len"
            ]  # This should be different for different tasks
            if metrics["success"] < 0.1:
                metrics[f"extra/{object_type}/eps_len_failed"] = metrics["eps_len"]
            else:
                metrics[f"extra/{object_type}/eps_len_success"] = metrics["eps_len"]

        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            print(traceback.format_exc())

        return metrics

    def calc_pickup_success(self, task, object_type):
        observations = task.get_observation_history()
        if object_type == "task_relevant":
            pickup_success = [obs["target_obj_was_pickedup"].item() for obs in observations]
        elif object_type == "any":
            pickup_success = [obs["an_object_is_in_hand"].item() for obs in observations]
        else:
            raise NotImplementedError
        pickup_success = sum(pickup_success) > 0
        return pickup_success

    def calculate_metrics(
        self,
        task: AbstractSPOCTask,
        all_actions: List[str],
        success: bool,
        additional_metrics: Dict[str, Any],
    ):
        metrics = {}

        metrics["eps_len"] = len(all_actions)
        metrics["success"] = float(success) + 1e-8
        if success:
            metrics["eps_len_succ"] = metrics["eps_len"]
        else:
            metrics["eps_len_fail"] = metrics["eps_len"]

        if "synsets" in task.task_info and len(task.task_info["synsets"]) == 1:
            metrics = self.get_extra_per_obj_metrics(task, metrics)

        if not success and (
            task.task_info["task_type"].startswith("Pickup")
            or task.task_info["task_type"].startswith("Fetch")
        ):
            metrics["failed_but_tried_pickup"] = int(THORActions.pickup in all_actions)

        trajectory = [obs["last_agent_location"][:3] for obs in task.observation_history]

        if task.room_poly_map is not None:
            percentage_visited, total_visited = calc_trajectory_room_visitation(
                task.room_poly_map, trajectory
            )
        else:
            percentage_visited, total_visited = 0, 0

        metrics["percentage_rooms_visited"] = percentage_visited
        metrics["total_rooms_visited"] = total_visited

        if "synsets" in task.task_info:
            list_of_object_types = task.task_info["synsets"]
            list_of_object_types = sorted(list_of_object_types)
            metrics["for_video_table/object_types"] = str(list_of_object_types)
            metrics["for_video_table/vis_pix_navigation"] = self.get_num_pixels_visible("nav", task)
            metrics["for_video_table/vis_pix_manipulation"] = self.get_num_pixels_visible(
                "manip", task
            )
            metrics["for_video_table/total_rooms"] = len(task.house["rooms"])
            metrics["for_video_table/pickup_sr"] = self.calc_pickup_success(
                task, object_type="task_relevant"
            )
            metrics["for_video_table/pickup_sr_any"] = self.calc_pickup_success(
                task, object_type="any"
            )
            metrics["for_video_table/has_agent_been_in_room"] = self.has_agent_been_in_obj_room(
                task
            )

        assert (
            len([k for k in additional_metrics.keys() if k in metrics]) == 0
        ), "You should not redefine metrics or have duplicates"
        metrics = {**metrics, **additional_metrics}

        return metrics

    def distribute_evaluate(
        self, agent: AbstractAgent, tasks_queue: mp.Queue, results_queue: mp.Queue
    ):
        verbose = platform.system() == "Darwin"

        send_videos_back = True

        self.task_sampler.task_spec_sampler = TaskSpecQueue(tasks_queue)

        num_tasks = 0
        while True:
            try:
                task = self.task_sampler.next_task()

                #print('Evaluating task:', task.task_info)

                if self.pre_defined_max_steps == -1:
                    task.max_steps = MAX_EPISODE_LEN_PER_TASK[task.task_info["task_type"]]
                else:
                    print(
                        f"IMPORTANT WARNING: YOU ARE SETTING MAX STEPS {self.pre_defined_max_steps} MANUALLY"
                        f"\nTASK {task.task_info['task_type']} REQUIRES"
                        f" {MAX_EPISODE_LEN_PER_TASK.get(task.task_info['task_type'], 'Not found')}"
                    )
                    task.max_steps = self.pre_defined_max_steps

            except EmptyQueueError:
                print(f"Terminating worker {self.worker_id}: No houses left in house_tasks.")
                break

            if verbose:
                print(f"Sample {num_tasks}")

            sample_result = self.evaluate_on_task(task=task, agent=agent, worker_id=self.worker_id)

            task_info = {**task.task_info, **task.task_info["eval_info"]}
            del task_info["eval_info"]

            to_log = dict(
                iter=num_tasks,
                task_type=task_info["task_type"],
                worker_id=self.worker_id,
                sample_id=task_info["sample_id"],
                metrics=sample_result["metrics"],
            )
            if verbose:
                print(to_log)

            video_table_data = None
            if send_videos_back and task_info["needs_video"]:
                eps_name = (
                    task_info["sample_id"] + "_" + sample_result["goal"].replace(" ", "-") + ".mp4"
                )

                video_path_to_send = cast(str, os.path.join(self.outdir, eps_name))
                print(f"Saving video to {video_path_to_send}")
                save_frames_to_mp4(
                    frames=sample_result["all_video_frames"], file_path=video_path_to_send, fps=5
                )

                topdown_view_path = os.path.join(self.outdir, eps_name + "_topdown.png")
                plt.imsave(fname=cast(str, topdown_view_path), arr=sample_result["top_down_frame"])

                # task_path = task_dict["task_path"]
                gt_episode_len = task_info["expert_length"]

                video_table_data = dict(
                    goal=sample_result["goal"],
                    video_path=video_path_to_send,
                    topdown_view_path=topdown_view_path,
                    success=bool(sample_result["metrics"]["success"] > 0.1),
                    eps_len=sample_result["metrics"]["eps_len"],
                    total_rooms_visited=sample_result["metrics"]["total_rooms_visited"],
                    gt_episode_len=gt_episode_len,
                    task_path=sample_result["task_path"],
                )
                video_table_data = {
                    **video_table_data,
                    **{
                        k.replace("for_video_table/", ""): v
                        for k, v in sample_result["metrics"].items()
                        if k.startswith("for_video_table/")
                    },
                }

            results_queue.put((to_log, video_table_data))
            num_tasks += 1

        print(f"Worker {self.worker_id} processed {num_tasks} tasks")


def load_objaverse_houses():
    # if self.eval_subset in ["val", "minival"]:
    #     subset_to_load = "val"
    # else:
    #     subset_to_load = self.eval_subset
    subset_to_load = "val"

    max_houses_per_split = {"train": 0, "val": 0, "test": 0}

    max_houses_per_split[subset_to_load] = int(1e9)
    return prior.load_dataset(
        dataset="spoc-data",
        entity="spoc-robot",
        revision="local-objaverse-procthor-houses",
        path_to_splits=None,
        split_to_path={
            k: os.path.join('/home/bera/Desktop/Codes/SPOC/spoc-robot-training/Evaluation/objaverse_houses', f"{k}.jsonl.gz")
            for k in ["train", "val", "test"]
        },
        max_houses_per_split=max_houses_per_split,
    )[subset_to_load]


def get_eval_run_name(args):
    exp_name = ["OnlineEval-revision-{}".format(args.benchmark_revision)]

    if args.extra_tag != "":
        exp_name.append(f"extra_tag={args.extra_tag}")

    if args.ckptStep is not None:
        exp_name.append(f"ckptStep={args.ckptStep}")

    exp_name.extend(
        [
            f"training_run_id={args.training_run_id}",
            f"eval_dataset={args.dataset_type}",
            f"eval_subset={args.eval_subset}",
            f"shuffle={args.shuffle}",
            f"sampling={args.sampling}",
        ]
    )

    return "-".join(exp_name)


def beta_exp_saturating(t, horizon, beta_min, beta_max, k=6.0):
    u = min(max(t / max(horizon, 1), 0.0), 1.0)
    g = (1.0 - math.exp(-k * u)) / (1.0 - math.exp(-k))
    return beta_min + (beta_max - beta_min) * g



# def forward_propagate(env, initial_state, current_t, first_action, horizon, q_net):
#     regular_actions = ['m', 'b', 'l', 'r', 'ls', 'rs']
#     start_time = time.time()
#     env.init_loc = initial_state
#     env.reset()
#     end_time = time.time()
#     print(f"Environment reset took {end_time - start_time:.4f} seconds")

#     t = current_t + 1
#     #print('first action:', first_action)
#     current_s, _, _ = env.step(first_action)
#     done = False

#     state_trajectory = [current_s]

#     #print('device:', next(q_net.parameters()).device)
#     start_time = time.time()
#     while t < horizon and not done:
#         state_norm = normalize_state(env, current_s, device = device)
#         t_norm = float(t) / float(max(horizon-1, 1))
#         t_tensor = torch.tensor([[t_norm]], dtype=torch.float32, device=device)  # (1,1)
#         x = torch.cat([state_norm, t_tensor], dim=1) 
#         Q_values = q_net(x)

#         a_idx = Q_values.argmax(dim=1).item()
#         action = regular_actions[a_idx]

#         next_s, r, done = env.step(action)

#         t += 1
#         current_s = next_s
#         state_trajectory.append(current_s)

#     end_time = time.time()
#     print(f"Forward propagation for horizon {horizon} took {end_time - start_time:.4f} seconds")

#     return state_trajectory


def forward_propagate(env, initial_state, current_t, first_action, horizon, q_net):
    regular_actions = ['m', 'b', 'l', 'r', 'ls', 'rs']
    agent = DiscreteUnicycleDynamics(x=initial_state[0], y=initial_state[1], theta=initial_state[2])

    t = current_t + 1
    #print('first action:', first_action)
    current_s = agent.update(first_action)
    done = False

    state_trajectory = [current_s]

    #print('device:', next(q_net.parameters()).device)
    start_time = time.time()

    q_net.eval()
    while t < horizon and not done:
        state_norm = normalize_state(env, current_s, device = device)
        t_norm = float(t) / float(max(horizon-1, 1))
        t_tensor = torch.tensor([[t_norm]], dtype=torch.float32, device=device)  # (1,1)
        x = torch.cat([state_norm, t_tensor], dim=1) 
        Q_values = q_net(x)

        a_idx = Q_values.argmax(dim=1).item()
        action = regular_actions[a_idx]

        next_s = agent.update(action)

        t += 1
        current_s = next_s
        state_trajectory.append(current_s)

    end_time = time.time()
    print(f"Forward propagation for horizon {horizon} took {end_time - start_time:.4f} seconds")

    return state_trajectory


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    args = parse_args()
    if args.wandb_logging:
        os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)

    gpu_devices = ["cpu"]
    if args.gpu_devices is not None and len(args.gpu_devices) > 0:
        gpu_devices = [int(device) for device in args.gpu_devices]

    if args.wandb_logging:
        assert (
            args.wandb_entity_name != "" and args.wandb_project_name != ""
        ), "wandb_entity_name and wandb_project_name must be provided"
        api = wandb.Api()
        run = api.run(f"{args.wandb_entity_name}/{args.wandb_project_name}/{args.training_run_id}")
    else:
        run = LoadLocalWandb(run_id=args.training_run_id, save_dir=args.local_checkpoint_dir)

    training_run_name = run.config["exp_name"]
    print('training_run_name:', args.training_run_id)
    eval_run_name = 'eval-' + training_run_name
    exp_base_dir = os.path.join(args.output_basedir, eval_run_name)
    ckpt_dir = os.path.join(exp_base_dir, "ckpts")
    exp_dir = os.path.join(exp_base_dir, datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S_%f"))
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)

    if args.wandb_logging:
        if args.ckptStep is None:
            raise ValueError("ckptStep is None")
        assert (
            args.wandb_entity_name != "" and args.wandb_project_name != ""
        ), "wandb_entity_name and wandb_project_name must be provided"
        ckpt_fn = f"{args.wandb_entity_name}/{args.wandb_project_name}/ckpt-{args.training_run_id}-{args.ckptStep}:latest"
        artifact = api.artifact(ckpt_fn)
        artifact.download(ckpt_dir)
        ckpt_pth = os.path.join(ckpt_dir, "model.ckpt")
    else:
        ckpt_pth = run.get_checkpoint(ckpt_step=args.ckptStep)

    model = run.config["model"]
    model_input_sensors = run.config["input_sensors"]
    if args.input_sensors is not None:
        # some sensors (e.g rooms_seen, room_current_seen) that are need to create model
        # are self-predicted and may not be provided to the agent as input
        assert set(args.input_sensors).issubset(
            set(model_input_sensors)
        ), f"{set(args.input_sensors)} is not a subset of {set(model_input_sensors)}"

    model_version = run.config["model_version"]

    if args.model_version_override != "auto":
        print(f"Enforcing model_version {args.model_version_override}")
        model_version = args.model_version_override

    loss = run.config["loss"]

    agent_class = REGISTERED_MODELS[model]
    agent_input = dict(
        model_version=model_version,
        input_sensors=model_input_sensors,
        loss=loss,
        sampling=args.sampling,
        ckpt_pth=ckpt_pth,
    )

    eval_run_name = get_eval_run_name(args)
    exp_base_dir = os.path.join(args.output_basedir, eval_run_name)
    ckpt_dir = os.path.join(exp_base_dir, "ckpts")
    exp_dir = os.path.join(exp_base_dir, datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S_%f"))
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)

    ##for double-det models:
    input_sensors=["raw_navigation_camera", "raw_manipulation_camera", "last_actions", "an_object_is_in_hand", "nav_task_relevant_object_bbox", "manip_task_relevant_object_bbox", "nav_accurate_object_bbox", "manip_accurate_object_bbox"]

    #for other models (without bounding box):
    #input_sensors=["raw_navigation_camera", "raw_manipulation_camera", "last_actions", "an_object_is_in_hand"]

    logging_sensor = VideoLogging()
    
    #start the worker:
    worker_args = {
    "gpu_device": 0,
    "houses": load_objaverse_houses(),
    "max_eps_len": 300,
    "input_sensors": input_sensors,
    "skip_done": False,
    "logging_sensor": logging_sensor,
    "outdir": exp_dir,
    "worker_id": 0,
    "det_type": "gt",
    }

    #create the worker:
    worker = OnlineEvaluatorWorker(**worker_args)

    # tasks_queue = mp.Queue()
    # results_queue = mp.Queue()

    #go to an alarm clock
    # task = {'sample_id': 'task=ObjectNavType,house=13653,sub_house_id=127', 'house_id': '013653', 'task_type': 'ObjectNavType', 'sub_house_id': 127, 'needs_video': True, 'raw_navigation_camera': '', 'sensors_path': '', 
    #         'observations': {'goal': 'find an alarm clock', 'initial_agent_location': np.array([ 4.19999981,  0.90099216,  5. ,0. , 90. ,0.]), 'actions': [], 'time_ids': [], 
    #         'templated_task_type': '{"task_type": "ObjectNavType", "house_index": 13653, "agent_starting_position": [4.199999809265137, 0.9009921550750732, 5.0], "agent_y_rotation": 90.0, "expert_length_bucket": "short", "expert_length": 30, "broad_synset_to_object_ids": {"alarm_clock.n.01": ["AlarmClock|4|5"]}, "synset_to_object_ids": {"alarm_clock.n.01": ["AlarmClock|4|5"]}, "synsets": ["alarm_clock.n.01"], "extras": {"chosen_object_id": "AlarmClock|4|5"}, "natural_language_spec": "find an alarm clock", "task_path": "/net/nfs.cirrascale/prior/datasets/vida_datasets/object_nav_v3_benchmark/ObjectNavType/val/013653/raw_navigation_camera__0.mp4", "hypernyms": ["instrument.n.01"], "freqs": [15]}'}}
    
    # #find an alarm clock in house 1
    # task = {'sample_id': 'task=ObjectNavType,house=1,sub_house_id=127', 'house_id': '1', 'task_type': 'ObjectNavType', 'sub_house_id': 127, 'needs_video': True, 'raw_navigation_camera': '', 'sensors_path': '', 
    #         'observations': {'goal': 'find an alarm clock', 'initial_agent_location': np.array([7.,  0.90099216,  5. ,0. , 90. ,0.]), 'actions': [], 'time_ids': [], 
    #         'templated_task_type': '{"task_type": "ObjectNavType", "house_index": 1, "agent_starting_position": [7.0, 0.9009921550750732, 5.0], "agent_y_rotation": 90.0, "expert_length_bucket": "short", "expert_length": 30, "broad_synset_to_object_ids": {"alarm_clock.n.01": ["AlarmClock|7|50"]}, "synset_to_object_ids": {"alarm_clock.n.01": ["AlarmClock|7|50"]}, "synsets": ["alarm_clock.n.01"], "extras": {"chosen_object_id": "AlarmClock|7|50"}, "natural_language_spec": "find an alarm clock", "task_path": "/net/nfs.cirrascale/prior/datasets/vida_datasets/object_nav_v3_benchmark/ObjectNavType/val/013653/raw_navigation_camera__0.mp4", "hypernyms": ["instrument.n.01"], "freqs": [15]}'}}

    # #find a television in house 1
    # task = {'sample_id': 'task=ObjectNavType,house=1,sub_house_id=127', 'house_id': '1', 'task_type': 'ObjectNavType', 'sub_house_id': 127, 'needs_video': True, 'raw_navigation_camera': '', 'sensors_path': '', 
    #         'observations': {'goal': 'find a television', 'initial_agent_location': np.array([1.,  0.90099216,  4. ,0. , 90. ,0.]), 'actions': [], 'time_ids': [], 
    #         'templated_task_type': '{"task_type": "ObjectNavType", "house_index": 1, "agent_starting_position": [1.0, 0.9009921550750732, 4.0], "agent_y_rotation": 90.0, "expert_length_bucket": "short", "expert_length": 30, "broad_synset_to_object_ids": {"television_receiver.n.01": ["television|6|4|1"]}, "synset_to_object_ids": {"television_receiver.n.01": ["television|6|4|1"]}, "synsets": ["television_receiver.n.01"], "extras": {"chosen_object_id": "television|6|4|1"}, "natural_language_spec": "find a television", "task_path": "/net/nfs.cirrascale/prior/datasets/vida_datasets/object_nav_v3_benchmark/ObjectNavType/val/013653/raw_navigation_camera__0.mp4", "hypernyms": ["instrument.n.01"], "freqs": [15]}'}}
    
    # #find a television in house 0
    # task = {'sample_id': 'task=ObjectNavType,house=0,sub_house_id=127', 'house_id': '0', 'task_type': 'ObjectNavType', 'sub_house_id': 127, 'needs_video': True, 'raw_navigation_camera': '', 'sensors_path': '', 
    #     'observations': {'goal': 'find a television', 'initial_agent_location': np.array([3.,  0.90099216,  6. ,0. , 90. ,0.]), 'actions': [], 'time_ids': [], 
    #     'templated_task_type': '{"task_type": "ObjectNavType", "house_index": 0, "agent_starting_position": [3.0, 0.9009921550750732, 6.0], "agent_y_rotation": 90.0, "expert_length_bucket": "short", "expert_length": 30, "broad_synset_to_object_ids": {"television_receiver.n.01": ["television|7|0|1"]}, "synset_to_object_ids": {"television_receiver.n.01": ["television|7|0|1"]}, "synsets": ["television_receiver.n.01"], "extras": {"chosen_object_id": "television|7|0|1"}, "natural_language_spec": "find a television", "task_path": "/net/nfs.cirrascale/prior/datasets/vida_datasets/object_nav_v3_benchmark/ObjectNavType/val/013653/raw_navigation_camera__0.mp4", "hypernyms": ["instrument.n.01"], "freqs": [15]}'}}


    # #pick up an alarm clock
    # task = {'sample_id': 'task=ObjectNavType,house=13653,sub_house_id=127', 'house_id': '013653', 'task_type': 'FetchType', 'sub_house_id': 127, 'needs_video': True, 'raw_navigation_camera': '', 'sensors_path': '', 
    #         'observations': {'goal': 'locate an alarm clock and pick up that alarm clock', 'initial_agent_location': np.array([ 4.19999981,  0.90099216,  5. ,0. , 90. ,0.]), 'actions': [], 'time_ids': [], 
    #         'templated_task_type': '{"task_type": "FetchType", "house_index": 13653, "agent_starting_position": [4.199999809265137, 0.9009921550750732, 5.0], "agent_y_rotation": 90.0, "expert_length_bucket": "short", "expert_length": 30, "broad_synset_to_object_ids": {"alarm_clock.n.01": ["AlarmClock|4|5"]}, "synset_to_object_ids": {"alarm_clock.n.01": ["AlarmClock|4|5"]}, "synsets": ["alarm_clock.n.01"], "extras": {"chosen_object_id": "AlarmClock|4|5"}, "natural_language_spec": "locate an alarm clock and pick up that alarm clock", "task_path": "/net/nfs.cirrascale/prior/datasets/vida_datasets/object_nav_v3_benchmark/ObjectNavType/val/013653/raw_navigation_camera__0.mp4", "hypernyms": ["instrument.n.01"], "freqs": [15]}'}}

    # #go to a television
    # task = {'sample_id': 'task=ObjectNavType,house=0,sub_house_id=127', 'house_id': '0', 'task_type': 'ObjectNavType', 'sub_house_id': 127, 'needs_video': True, 'raw_navigation_camera': '', 'sensors_path': '', 
    #         'observations': {'goal': 'go to a television', 'initial_agent_location': np.array([ 3.25,  0.90099216,  5.75,0. , 90. ,0.]), 'actions': [], 'time_ids': [], 
    #         'templated_task_type': '{"task_type": "ObjectNavType", "house_index": 0, "agent_starting_position": [3.25, 0.9009921550750732, 5.75], "agent_y_rotation": 90.0, "expert_length_bucket": "short", "expert_length": 30, "broad_synset_to_object_ids": {"television_receiver.n.01": ["television|7|0|1", "television|7|0|0"]}, "synset_to_object_ids": {"television_receiver.n.01": ["television|7|0|1", "television|7|0|1"]}, "synsets": ["television_receiver.n.01"], "extras": {"chosen_object_id": "television|7|0|1"}, "natural_language_spec": "go to a television", "task_path": "/net/nfs.cirrascale/prior/datasets/vida_datasets/object_nav_v3_benchmark/ObjectNavType/val/013653/raw_navigation_camera__0.mp4", "hypernyms": ["instrument.n.01"], "freqs": [15]}'}}
    
    # #go to a pan
    # task = {'sample_id': 'task=ObjectNavType,house=0,sub_house_id=127', 'house_id': '0', 'task_type': 'ObjectNavType', 'sub_house_id': 127, 'needs_video': False, 'raw_navigation_camera': '', 'sensors_path': '', 
    #         'observations': {'goal': 'go to a pan', 'initial_agent_location': np.array([ 3.25,  0.90099216,  5.75,0. , 90. ,0.]), 'actions': [], 'time_ids': [], 
    #         'templated_task_type': '{"task_type": "ObjectNavType", "house_index": 0, "agent_starting_position": [3.25, 0.9009921550750732, 5.75], "agent_y_rotation": 90.0, "expert_length_bucket": "short", "expert_length": 30, "broad_synset_to_object_ids": {"pan.n.01": ["Pan|6|34"]}, "synset_to_object_ids": {"pan.n.01": ["Pan|6|34"]}, "synsets": ["pan.n.01"], "extras": {"chosen_object_id": "Pan|6|34"}, "natural_language_spec": "go to a pan", "task_path": "/net/nfs.cirrascale/prior/datasets/vida_datasets/object_nav_v3_benchmark/ObjectNavType/val/013653/raw_navigation_camera__0.mp4", "hypernyms": ["instrument.n.01"], "freqs": [15]}'}}


    #go to a kitchen
    # task = {'sample_id': 'task=RoomNav,house=6090,sub_house_id=127', 'house_id': '006090', 'task_type': 'RoomNav', 'sub_house_id': 127, 'needs_video': True, 'raw_navigation_camera': '', 
    #         'sensors_path': '', 'observations': {'goal': 'go to a kitchen', 'initial_agent_location': np.array([3.05000019,  0.90099216, 5.75, 0., 60.00000381, 0.]), 'actions': [], 'time_ids': [], 'templated_task_type': '{"task_type": "RoomNav", "house_index": 6090, "agent_starting_position": [10.050000190734863, 0.9009921550750732, 11.75], "agent_y_rotation": 60.000003814697266, "expert_length_bucket": "short", "expert_length": 17, "room_types": ["Kitchen"], "room_ids": {"Kitchen": ["room|8"]}, "extras": {"chosen_room_id": "room|8"}, "natural_language_spec": "go to a kitchen", "task_path": "/net/nfs.cirrascale/prior/datasets/vida_datasets/benchmark_Oct22_closed_type/RoomNav/val/006090/raw_navigation_camera__0.mp4", "freqs": [51]}'}}

    # #find a bowl in house 2:
    # task = {'sample_id': 'task=ObjectNavType,house=2,sub_house_id=2', 'house_id': '2', 'task_type': 'ObjectNavType', 'sub_house_id': 2, 'needs_video': True, 'raw_navigation_camera': '', 'sensors_path': '', 
    #         'observations': {'goal': 'go to a bowl', 'initial_agent_location': np.array([0.8,  0.90099216,  3.75 , 90. ,0.]), 'actions': [], 'time_ids': [], 
    #         'templated_task_type': '{"task_type": "ObjectNavType", "house_index": 2, "agent_starting_position": [0, 0.9009921550750732, 0], "agent_y_rotation": 90.0, "expert_length_bucket": "short", "expert_length": 30, "broad_synset_to_object_ids": {"bowl.n.03": ["Bowl|2|5"]}, "synset_to_object_ids": {"bowl.n.03": ["Bowl|2|5"]}, "synsets": ["bowl.n.03"], "extras": {"chosen_object_id": "Bowl|2|5"}, "natural_language_spec": "go to a bowl", "task_path": "/net/nfs.cirrascale/prior/datasets/vida_datasets/object_nav_v3_benchmark/ObjectNavType/val/013653/raw_navigation_camera__0.mp4", "hypernyms": ["instrument.n.01"], "freqs": [15]}'}}
    

    # #find a bowl in house 4:
    # task = {'sample_id': 'task=ObjectNavType,house=8,sub_house_id=4', 'house_id': '8', 'task_type': 'ObjectNavType', 'sub_house_id': 8, 'needs_video': True, 'raw_navigation_camera': '', 'sensors_path': '', 
    #         'observations': {'goal': 'go to a bowl', 'initial_agent_location': np.array([1.67,  0.90099216,  2.5 , 90. ,0.]), 'actions': [], 'time_ids': [], 
    #         'templated_task_type': '{"task_type": "ObjectNavType", "house_index": 4, "agent_starting_position": [0, 0.9009921550750732, 0], "agent_y_rotation": 90.0, "expert_length_bucket": "short", "expert_length": 30, "broad_synset_to_object_ids": {"bowl.n.03": ["Bowl|6|38"]}, "synset_to_object_ids": {"bowl.n.03": ["Bowl|6|38"]}, "synsets": ["bowl.n.03"], "extras": {"chosen_object_id": "Bowl|6|38"}, "natural_language_spec": "go to a bowl", "task_path": "/net/nfs.cirrascale/prior/datasets/vida_datasets/object_nav_v3_benchmark/ObjectNavType/val/013653/raw_navigation_camera__0.mp4", "hypernyms": ["instrument.n.01"], "freqs": [15]}'}}
    
    #find a bowl in house 30:
    task = {'sample_id': 'task=ObjectNavType,house=30,sub_house_id=2', 'house_id': '30', 'task_type': 'ObjectNavType', 'sub_house_id': 30, 'needs_video': True, 'raw_navigation_camera': '', 'sensors_path': '', 
            'observations': {'goal': 'go to a bowl', 'initial_agent_location': np.array([1.5,  0.90099216,  0.5 , 90. ,0.]), 'actions': [], 'time_ids': [], 
            'templated_task_type': '{"task_type": "ObjectNavType", "house_index": 30, "agent_starting_position": [0, 0.9009921550750732, 0], "agent_y_rotation": 90.0, "expert_length_bucket": "short", "expert_length": 30, "broad_synset_to_object_ids": {"bowl.n.03": ["Bowl|2|5"]}, "synset_to_object_ids": {"bowl.n.03": ["Bowl|2|5"]}, "synsets": ["bowl.n.03"], "extras": {"chosen_object_id": "Bowl|2|5"}, "natural_language_spec": "go to a bowl", "task_path": "/net/nfs.cirrascale/prior/datasets/vida_datasets/object_nav_v3_benchmark/ObjectNavType/val/013653/raw_navigation_camera__0.mp4", "hypernyms": ["instrument.n.01"], "freqs": [15]}'}}


    #load the Q-network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = select_model_file()
    ckpt = torch.load(model_path, map_location=device)
    #obs_dim = ckpt["obs_dim"]
    n_actions = ckpt["n_actions"]
    idx_to_action =['m', 'b', 'l', 'r', 'ls', 'rs']
 
    feat_dim = 4

    Q_net = QNetwork(input_dim=feat_dim + 1, n_actions=n_actions).to(device)
    Q_net.load_state_dict(ckpt["policy_state_dict"])
    Q_net.eval()

    STL_horizon = 60 #CHANGE LATER!

    #create 2D environment for Q-network:
    goals = {0: {'center': (5.2, 1.8), 'radius': 0.5, 'movement':{'type':'static'}}}

    targets = {}
    config = {
        'house_index': 30,
        'init_loc':[1.5, 0.5, 90], #initial location of the agent (x, y)
        "dt": 1,
        "render": False,
		'dt_render': 0.01,
		'goals': goals, #goal regions for the agent
        "obstacle_location": [100.0, 100.0],
        "obstacle_size": 0.0,
        "randomize_loc": False, #whether to randomize the agent location at the end of each episode
		'deterministic': False,
		'auto_entropy':True,
		"dynamics": "discrete unicycle", #dynamics model to use
		"targets": targets,
		"disturbance": None, #disturbance range in both x and y directions [w_min, w_max]
        "agent_as_point": True
    }

    env_2d = Continuous2DEnv(config)

    num_trials = 1
    num_success = 0
    total_eps_len = 0
    num_target_reached = 0

    starting_time = time.time()
    for _ in range(num_trials):
        target_reached = False
        tasks_queue = mp.Queue()
        results_queue = mp.Queue()

        print("Trial:", _+1)
        tasks_queue.put(task)

        start_worker(
            worker,
            agent_class,
            agent_input,
            device= 0,
            tasks_queue=tasks_queue,
            results_queue=results_queue,
        )

        results = results_queue.get()[0]
        success = results["metrics"]["success"]
        eps_len = results["metrics"]["eps_len"]
        
        total_eps_len += eps_len

        if target_reached:
            num_target_reached += 1

        if success == 1.00000001:
            print("The agent successfully completed the task!")
            num_success += 1

    finish_time = time.time()
    elapsed_time = finish_time - starting_time

    print('Number of successful trials:', num_success)
    print(f"Success rate over {num_trials} trials: {num_success/num_trials*100}%")
    print(f"Average episode length over {num_trials} trials: {total_eps_len/num_trials}")
    print('Total time elapsed:', elapsed_time, 'seconds')
    print("Target reached in", num_target_reached, "out of", num_trials, "trials.")



    # # Ensure the model can be loaded
    # agent_class.build_agent(**agent_input, device="cuda")

    # print("Agent built successfully, starting evaluation...")



    # task = ObjectNavTask(task_info={'task_type': 'ObjectNavType', 'house_index': '13653', 'num_rooms': 4, 'agent_starting_position': {'x': 4.199999809265137, 'y': 0.9009921550750732, 'z': 5.0}, 'agent_y_rotation': 90.0, 'natural_language_spec': 'go to an alarm clock', 'eval_info': {'sample_id': 'task=ObjectNavType,house=13653,sub_house_id=127', 'needs_video': True, 'task_type': 'ObjectNavType', 'house_index': 13653, 'agent_starting_position': [4.199999809265137, 0.9009921550750732, 5.0], 'agent_y_rotation': 90.0, 'expert_length_bucket': 'short', 'expert_length': 30, 'broad_synset_to_object_ids': {'alarm_clock.n.01': ['AlarmClock|4|5']}, 'synset_to_object_ids': {'alarm_clock.n.01': ['AlarmClock|4|5']}, 'synsets': ['alarm_clock.n.01'], 'extras': {'chosen_object_id': 'AlarmClock|4|5'}, 'natural_language_spec': 'go to an alarm clock', 'task_path': '/net/nfs.cirrascale/prior/datasets/vida_datasets/object_nav_v3_benchmark/ObjectNavType/val/013653/raw_navigation_camera__0.mp4', 'hypernyms': ['instrument.n.01'], 'freqs': [15]}, 'synsets': ['alarm_clock.n.01'], 'synset_to_object_ids': {'alarm_clock.n.01': ['AlarmClock|4|5']}, 'broad_synset_to_object_ids': {'alarm_clock.n.01': ['AlarmClock|4|5']}, 'extras': {}, 'followed_path': [{'x': 4.199999809265137, 'y': 0.9009921550750732, 'z': 5.0}], 'agent_poses': [{'name': 'agent', 'position': {'x': 4.199999809265137, 'y': 0.9009921550750732, 'z': 5.0}, 'rotation': {'x': -0.0, 'y': 90.0, 'z': 0.0}, 'cameraHorizon': 25.200002670288086, 'isStanding': False, 'inHighFrictionArea': False, 'arm': {'joints': [{'name': 'stretch_robot_lift_jnt', 'position': {'x': 4.12999963760376, 'y': 0.4152766466140747, 'z': 5.135000228881836}, 'rootRelativePosition': {'x': 0.04607595503330231, 'y': 0.32747650146484375, 'z': -0.2787679135799408}, 'rotation': {'x': -0.0, 'y': 1.0, 'z': -0.0, 'w': 180.0}, 'rootRelativeRotation': {'x': 1.0, 'y': 0.0, 'z': 0.0, 'w': 0.0}, 'localRotation': {'x': 1.0, 'y': 0.0, 'z': 0.0, 'w': 0.0}, 'armBaseHeight': None, 'elbowOrientation': None}, {'name': 'stretch_robot_arm_1_jnt', 'position': {'x': 4.093076229095459, 'y': 0.4152766466140747, 'z': 4.8802900314331055}, 'rootRelativePosition': {'x': 0.0829993188381195, 'y': 0.32747650146484375, 'z': -0.024057626724243164}, 'rotation': {'x': -0.0, 'y': 1.0, 'z': -0.0, 'w': 180.0}, 'rootRelativeRotation': {'x': 1.0, 'y': 0.0, 'z': 0.0, 'w': 0.0}, 'localRotation': {'x': 1.0, 'y': 0.0, 'z': 0.0, 'w': 0.0}, 'armBaseHeight': None, 'elbowOrientation': None}, {'name': 'stretch_robot_arm_2_jnt', 'position': {'x': 4.093076229095459, 'y': 0.4152766466140747, 'z': 4.8672895431518555}, 'rootRelativePosition': {'x': 0.0829993188381195, 'y': 0.32747650146484375, 'z': -0.011057138442993164}, 'rotation': {'x': -0.0, 'y': 1.0, 'z': -0.0, 'w': 180.0}, 'rootRelativeRotation': {'x': 1.0, 'y': 0.0, 'z': 0.0, 'w': 0.0}, 'localRotation': {'x': 1.0, 'y': 0.0, 'z': 0.0, 'w': 0.0}, 'armBaseHeight': None, 'elbowOrientation': None}, {'name': 'stretch_robot_arm_3_jnt', 'position': {'x': 4.093076229095459, 'y': 0.4152766466140747, 'z': 4.8542890548706055}, 'rootRelativePosition': {'x': 0.08299930393695831, 'y': 0.32747650146484375, 'z': 0.001943349838256836}, 'rotation': {'x': -0.0, 'y': 1.0, 'z': -0.0, 'w': 180.0}, 'rootRelativeRotation': {'x': 1.0, 'y': 0.0, 'z': 0.0, 'w': 0.0}, 'localRotation': {'x': 1.0, 'y': 0.0, 'z': 0.0, 'w': 0.0}, 'armBaseHeight': None, 'elbowOrientation': None}, {'name': 'stretch_robot_arm_4_jnt', 'position': {'x': 4.093076229095459, 'y': 0.4152766466140747, 'z': 4.841289043426514}, 'rootRelativePosition': {'x': 0.08299930393695831, 'y': 0.32747650146484375, 'z': 0.01494339108467102}, 'rotation': {'x': -0.0, 'y': 1.0, 'z': -0.0, 'w': 180.0}, 'rootRelativeRotation': {'x': 1.0, 'y': 0.0, 'z': 0.0, 'w': 0.0}, 'localRotation': {'x': 1.0, 'y': 0.0, 'z': 0.0, 'w': 0.0}, 'armBaseHeight': None, 'elbowOrientation': None}, {'name': 'stretch_robot_arm_5_jnt', 'position': {'x': 4.093076229095459, 'y': 0.4152766466140747, 'z': 4.829542636871338}, 'rootRelativePosition': {'x': 0.08299930393695831, 'y': 0.32747650146484375, 'z': 0.026689797639846802}, 'rotation': {'x': -0.0, 'y': 1.0, 'z': -0.0, 'w': 180.0}, 'rootRelativeRotation': {'x': 1.0, 'y': 0.0, 'z': 0.0, 'w': 0.0}, 'localRotation': {'x': 1.0, 'y': 0.0, 'z': 0.0, 'w': 0.0}, 'armBaseHeight': None, 'elbowOrientation': None}, {'name': 'stretch_robot_wrist_1_jnt', 'position': {'x': 4.1761474609375, 'y': 0.4017653465270996, 'z': 4.856293201446533}, 'rootRelativePosition': {'x': -7.193908095359802e-05, 'y': 0.31396520137786865, 'z': -6.085634231567383e-05}, 'rotation': {'x': 1.0, 'y': 0.0, 'z': 0.0, 'w': 0.0}, 'rootRelativeRotation': {'x': 0.0, 'y': 1.0, 'z': 0.0, 'w': 180.0}, 'localRotation': {'x': 0.0, 'y': 1.0, 'z': 0.0, 'w': 180.0}, 'armBaseHeight': None, 'elbowOrientation': None}, {'name': 'stretch_robot_wrist_2_jnt', 'position': {'x': 4.1761474609375, 'y': 0.35102641582489014, 'z': 4.856293201446533}, 'rootRelativePosition': {'x': -7.193908095359802e-05, 'y': 0.2632262706756592, 'z': -6.085634231567383e-05}, 'rotation': {'x': 1.0, 'y': 0.0, 'z': 0.0, 'w': 0.0}, 'rootRelativeRotation': {'x': 0.0, 'y': 1.0, 'z': 0.0, 'w': 180.0}, 'localRotation': {'x': 1.0, 'y': 0.0, 'z': 0.0, 'w': 0.0}, 'armBaseHeight': None, 'elbowOrientation': None}], 'heldObjects': [], 'pickupableObjects': [], 'touchedNotHeldObjects': [], 'handSphereCenter': {'x': 4.1761474609375, 'y': 0.25239962339401245, 'z': 5.060993194580078}, 'handSphereRadius': 0.05999999865889549}}], 'taken_actions': [], 'action_successes': [], 'id': 'ObjectNavType_13653_1763743654_gotoanalarmclock'})

    # houses = load_objaverse_houses()
    # print(houses)
    # gpu_device = gpu_devices[0]
    # worker_id = 0
    # worker_args = {'gpu_device': 0, 'houses': houses, 'max_eps_len': -1, 'input_sensors': ['raw_navigation_camera', 'raw_manipulation_camera', 'last_actions', 'an_object_is_in_hand', 'nav_task_relevant_object_bbox', 'manip_task_relevant_object_bbox', 'nav_accurate_object_bbox', 'manip_accurate_object_bbox'],
    #       'skip_done': False, 'logging_sensor': <utils.visualization_utils.VideoLogging object at 0x74e9447740a0>,
    #       'outdir': 'tmp_log/OnlineEval-revision-chores-small-training_run_id=SigLIP-ViTb-3-double-det-CHORES-S-eval_dataset=-eval_subset=minival-shuffle=True-sampling=sample/11_21_2025_12_20_06_233293', 'worker_id': 0, 'det_type': 'gt'}


    #evaluate_on_task                                                                                                                                                                                       