##USING HJB APPROACH (4/10/2026)
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
import stlrom

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

from robustness_calculator import calculate_goal_predicate_robustness
from RL.src.networks import QNetwork
from RL.src.main import select_model_file, normalize_state
from RL.src.simulator import Continuous2DEnv
from RL.src.dynamics import DiscreteUnicycleDynamics
from RL.src.geometry import _swept_circle_clipped_translation

#from HJB import house_BRT
from HJB import BRT_subprocess

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
    parser.add_argument("--local_checkpoint_dir", default="/home/bera/Desktop/Codes/STL Aware Foundational Models/SPOC/spoc-robot-training/Evaluation/pre-trained")
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
        #global target_reached
        global stl_task_done

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
        main_task_done = False #whether FM thinks the main task is satisfied
        main_task_actually_done = False #whether the main task is actually done according to the task's definition of success (which may not be the same as FM's prediction of whether the main task is done)
        mismatch_episode = False

        normalized_q_list = []
        normalized_logit_list = []
        regular_Q_vals = []
        regular_logits = []

        init_t = 0
        eps_idx = init_t
        num_of_available_actions_per_step = []

        state_trajectory = [] + [(9999, 9999, 0)] * init_t #add very far away states so the robustness values would be too low (for reachability predicates) FIXME: this is a hack to handle the test cases where init_t != 0 but ideally we should start from 0

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
                    original_probs = torch.softmax(torch.tensor(logits), -1).detach().numpy()

                all_frames.append(curr_frame)
                

                if (action in ["end", "sub_done"]) and (not main_task_done):
                    if task.successful_if_done(strict_success=False):
                        main_task_actually_done = True

                    main_task_done = True #consider main task done according to FM even if it's not actually done according to the task's definition of success
                    print("FM: Main task satisifed at step", eps_idx)
                    main_task_done_step = eps_idx

                full_pose = task.controller.get_current_agent_full_pose() #(x, y, z) *y is height
                current_state = (full_pose['position']['x'], full_pose['position']['z'], full_pose['rotation']['y']) #(x, y, theta in degrees)
                state_trajectory.append(current_state)

                # if not stl_task_done:
                #     if eps_idx >= 80: #start checking STL satisfaction after 80 steps (since the second STL constraint is starts at 80)
                #         if len(state_trajectory) > STL_horizon:
                #             temp_trajectory = state_trajectory[:STL_horizon + 1] #get the first STL_horizon + 1 states in the trajectory to calculate the robustness values for the STL predicates (since STL specifications are defined over a finite horizon)
                #             predicate_signals = np.ones((STL_horizon + 1, len(goals))) * -999 
                #         else:
                #             temp_trajectory = state_trajectory
                #             predicate_signals = np.ones((len(state_trajectory), len(goals))) * -999 #initialize with very negative value (i.e., violation)


                #         for goal_id in goals.keys():
                #             g = goals[goal_id]
                #             predicate_robustness = calculate_goal_predicate_robustness(temp_trajectory, g['center'], g['radius'])
                #             predicate_signals[:, goal_id] = predicate_robustness

                #         current_task_robustness = get_task_robustness(stl_task_str, predicate_signals) #calculate the overall task robustness degree for the current trajectory
                #         if current_task_robustness > 0:
                #             stl_task_done = True
                #             print("STL task satisifed at step", eps_idx)
                

                if np.linalg.norm(np.array(current_state[:2]) - np.array(goals[0]['center'])) < goals[0]['radius']:
                    print("Goal 0 reached at step", eps_idx)
                    stl_task_done = True

                if np.linalg.norm(np.array(current_state[:2]) - np.array(goals[1]['center'])) < goals[1]['radius']:
                    print("Goal 1 reached at step", eps_idx)
                    stl_task_done = True

                # if not stl_task_done: #modify the distribution if STL task is not yet satisfied
                #     regular_actions = ['m', 'b', 'l', 'r', 'ls', 'rs']
                #     regular_action_logits = np.array([logits[action_list.index(a)] for a in regular_actions])
                #     p = torch.softmax(torch.tensor(regular_action_logits, dtype=torch.float32), -1).detach().numpy() #original action distribution from FM



                    #print('original action distribution:', p)

                if eps_idx <= time_horizon:
                    #brt_value =  house_BRT.get_brt_value_at_time(grid, all_brt_values, times, current_state, time_to_go=20 - eps_idx)
                    # brt_value = get_brt_value_at_time_numpy(brt_data, current_state, time_to_go = time_horizon - eps_idx)
                    # print("current state:", current_state)
                    # print("current BRT value:", brt_value)

                    regular_actions = ['m', 'b', 'l', 'r', 'ls', 'rs']
                    regular_action_logits = np.array([logits[action_list.index(a)] for a in regular_actions])
                    p = torch.softmax(torch.tensor(regular_action_logits, dtype=torch.float32), -1).detach().numpy() #original action distribution from FM

                    v = np.zeros((len(regular_actions),), dtype=np.float64)
                    safe_actions = []
                    #print("current state:", current_state)
                    for i, a in enumerate(regular_actions):
                        next_state, _, _ = unicycle_step(current_state, a, dt=1.0) #get the next state after taking the action for 1 second according to unicycle dynamics
                        #print('next state after action {}: {}'.format(a, next_state))
                        next_brt_value = get_brt_value_at_time_numpy(brt_data, next_state, time_to_go = time_horizon - eps_idx - 1)
                        v[i] = (next_brt_value < 0) #safe actions will have value 1, unsafe actions will have value 0 since the BRT value is negative inside the BRT and positive outside the BRT
                        if v[i] > 0.5:
                            safe_actions.append(a)

                    print("safe actions according to BRT: {} at state {}".format(safe_actions, current_state))
                    safe = (v > 0.5)

                    q = np.zeros_like(p)
                    mass = float(p[safe].sum())
                    if mass > 0:
                        q[safe] = p[safe] / mass #re-normalize the probabilities of the safe actions
                    else:
                        print("Warning: no safe actions according to BRT, using original distribution")
                        q = p #if there are no safe actions, use the original distribution

                    new_probs = torch.tensor(q, dtype=torch.float32)
                    #print('New action distribution:', new_probs)
                    #print('Distribution modified:' , not np.allclose(p, q))
                    #print("q.sum =", q.sum())

                    action_idx = torch.distributions.categorical.Categorical(probs=new_probs).sample() #sample the action on the modified logits
                    action = regular_actions[action_idx]

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

                if main_task_done and stl_task_done:
                    break
        

        if display_realtime:
            cv2.destroyAllWindows()
        
        # if main_task_done:
        #     action = 'end' #take a pseudo "end" action to end the episode if main task is done but not ended due to skip_done setting, so that the final success metrics would be calculated correctly based on the task's definition of success 
        #     task.step_with_action_str(action)
        
        if main_task_done:
            print(f'Main task satisfied at step {main_task_done_step}, breaking out of the loop.')

        success = main_task_actually_done

        print('Main task success accordign to FM:', main_task_done)
        print("Main task success:", main_task_actually_done)
        print('STL task success:', stl_task_done)

        target_ids = None
        if "synset_to_object_ids" in task.task_info:
            target_ids = list(
                chain.from_iterable(task.task_info.get("synset_to_object_ids", None).values())
            )

        #print("Path:", task.task_info["followed_path"])


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

        #add mismatch_episode flag to metrics so that we can analyze the mismatch cases separately
        metrics["mismatch_episode"] = mismatch_episode

        print("All actions taken:", all_actions)

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
        #metrics["success"] = float(success) + 1e-8
        metrics["success"] = success
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
            k: os.path.join('/home/bera/Desktop/Codes/STL Aware Foundational Models/SPOC/spoc-robot-training/Evaluation/objaverse_houses', f"{k}.jsonl.gz")
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


    

#HELPER FUNCTIONS TO CHECK WHETHET TWO STATES ARE ALMOST THE SAME:
def _wrap_angle_deg(a: float) -> float:
    """Map angle to [0, 360)."""
    return float(a) % 360.0

def _angle_diff_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two headings in degrees."""
    a = _wrap_angle_deg(a)
    b = _wrap_angle_deg(b)
    d = abs(a - b)
    return min(d, 360.0 - d)

def states_almost_same(
    thor_state,              # (x, z, yaw_deg)
    pred_state,              # (x, y, yaw_deg) or np.array([x, y, yaw])
    *,
    pos_tol: float = 1e-3,   # meters (start with 1e-3 to 1e-2)
    yaw_tol: float = 1e-2,   # degrees (start with 1e-2 to 1e-1)
) -> tuple[bool, dict]:
    """
    Returns (ok, info) where info has the position error and yaw error.
    Assumes your 2D map uses (x, y) where y corresponds to THOR z.
    """
    tx, tz, tyaw = map(float, thor_state)

    p = np.asarray(pred_state, dtype=np.float64).reshape(-1)
    px, py, pyaw = float(p[0]), float(p[1]), float(p[2])

    pos_err = float(np.hypot(tx - px, tz - py))
    yaw_err = float(_angle_diff_deg(tyaw, pyaw))

    ok = (pos_err <= pos_tol) and (yaw_err <= yaw_tol)
    return ok, {"pos_err": pos_err, "yaw_err": yaw_err}



def get_task_robustness(task, signals):
    stl_driver =stlrom.STLDriver()
    stl_driver.parse_string(task)

    #add the samples:
    for i in range(signals.shape[0]):
        sample = [i] + signals[i].tolist()
        stl_driver.add_sample(sample)

    phi = stl_driver.get_monitor("phi") #overall task
    robustness = phi.eval_rob()

    return robustness    

def remove_objects_by_id(house_dict, ids_to_remove):
    ids_to_remove = set(ids_to_remove)
    house_dict = dict(house_dict)  # shallow copy

    new_objs = []
    for o in house_dict.get("objects", []):
        if o.get("id") in ids_to_remove:
            continue
        new_objs.append(o)

    house_dict["objects"] = new_objs
    return house_dict

def remove_windows_by_id(house_dict, ids_to_remove):
    ids_to_remove = set(ids_to_remove)
    house_dict = dict(house_dict)  # shallow copy
    house_dict["windows"] = [w for w in house_dict.get("windows", [])
                             if w.get("id") not in ids_to_remove]
    return house_dict

def get_brt_value_at_time_numpy(brt_data, state, time_to_go):
    """
    Get interpolated BRT value using numpy only.
    
    Args:
        brt_data: dict with 'times', 'all_brt_values', 'coordinate_vectors'
        state: [x, y, theta] where theta is in DEGREES
        time_to_go: remaining time to reach target
    
    Returns:
        Interpolated value (negative = inside BRT, positive = outside)
    """
    from scipy.ndimage import map_coordinates
    
    times = brt_data["times"]
    all_brt_values = brt_data["all_brt_values"]
    coord_vectors = brt_data["coordinate_vectors"]
    
    state = np.asarray(state, dtype=np.float64)
    state[2] = np.deg2rad(state[2])  # Convert theta to radians
    
    # Time index
    times_flipped = times[::-1]
    indices_flipped = np.arange(len(times))[::-1]
    query_t = -time_to_go
    time_idx = np.interp(query_t, times_flipped, indices_flipped)
    
    # Spatial indices
    indices = [time_idx]
    for i in range(3):
        coord_vec = coord_vectors[i]
        lo, hi = coord_vec[0], coord_vec[-1]
        n = len(coord_vec)
        
        if i == 2:  # theta periodic
            s = state[i] % (2 * np.pi)
        else:
            s = state[i]
        
        idx = (s - lo) / (hi - lo) * (n - 1)
        indices.append(idx)
    
    indices = np.array(indices).reshape(-1, 1)
    
    value = map_coordinates(
        all_brt_values,
        indices,
        order=1,
        mode='wrap'
    )
    
    return float(value[0])


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

    houses_lazy = load_objaverse_houses()
    houses = list(houses_lazy)          # materialize into a normal list of dicts

    houses[9] = remove_objects_by_id(houses[9], ["ObjaFoldingChair|2|2"])
    houses[9] = remove_windows_by_id(houses[9], ["window|2|1"])

    #remove stuff from house-152
    houses[152] = remove_objects_by_id(houses[152], ["FloorLamp|3|1"])
    houses[152] = remove_objects_by_id(houses[152], ["ObjaWheelchair|2|3"])
    houses[152] = remove_objects_by_id(houses[152], ["ObjaTrunk|3|3"])
    houses[152] = remove_objects_by_id(houses[152], ["chair-diningtable-2|2|2|2"])
    houses[152] = remove_objects_by_id(houses[152], ["Bowl|3|30"])
    houses[152] = remove_objects_by_id(houses[152], ['SideTable|2|4'])

    houses[143] = remove_objects_by_id(houses[143], ["ObjaMailbox|2|3"])
    
    #start the worker:
    worker_args = {
    "gpu_device": 0,
    "houses": houses,
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


    #load the Q-network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx_to_action =['m', 'b', 'l', 'r', 'ls', 'rs']
 
    feat_dim = 4



    ##########################################
    # HOUSE 152 GOALS AND TASKS FOR TESTING:
    ##########################################
    task = {'sample_id': 'task=ObjectNavType,house=152,sub_house_id=152', 'house_id': '152', 'task_type': 'ObjectNavType', 'sub_house_id': 152, 'needs_video': True, 'raw_navigation_camera': '', 'sensors_path': '', 
            'observations': {'goal': 'go to a bowl', 'initial_agent_location': np.array([7,  0.90099216,  2 , 270. ,0.]), 'actions': [], 'time_ids': [], 
            'templated_task_type': '{"task_type": "ObjectNavType", "house_index": 152, "agent_starting_position": [0, 0.9009921550750732, 0], "agent_y_rotation": 90.0, "expert_length_bucket": "short", "expert_length": 30, "broad_synset_to_object_ids": {"bowl.n.03": ["Bowl|2|5"]}, "synset_to_object_ids": {"bowl.n.03": ["Bowl|2|5"]}, "synsets": ["bowl.n.03"], "extras": {"chosen_object_id": "Bowl|2|5"}, "natural_language_spec": "go to a bowl", "task_path": "/net/nfs.cirrascale/prior/datasets/vida_datasets/object_nav_v3_benchmark/ObjectNavType/val/013653/raw_navigation_camera__0.mp4", "hypernyms": ["instrument.n.01"], "freqs": [15]}'}}
    

    # #go to a cellphone in house 152:
    # task = {'sample_id': 'task=ObjectNavType,house=152,sub_house_id=152', 'house_id': '152', 'task_type': 'ObjectNavType', 'sub_house_id': 152, 'needs_video': True, 'raw_navigation_camera': '', 'sensors_path': '', 
    #         'observations': {'goal': 'find a cellphone', 'initial_agent_location': np.array([3,  0.90099216,  6 , 180.0 ,0.]), 'actions': [], 'time_ids': [], 
    #         'templated_task_type': '{"task_type": "ObjectNavType", "house_index": 152, "agent_starting_position": [0, 0.9009921550750732, 0], "agent_y_rotation": 90.0, "expert_length_bucket": "short", "expert_length": 30, "broad_synset_to_object_ids": {"cellular_telephone.n.01": ["CellPhone|3|22"]}, "synset_to_object_ids": {"cellular_telephone.n.01": ["CellPhone|3|22"]}, "synsets": ["cellular_telephone.n.01"], "extras": {"chosen_object_id": "CellPhone|3|22"}, "natural_language_spec": "go to a television", "task_path": "/net/nfs.cirrascale/prior/datasets/vida_datasets/object_nav_v3_benchmark/ObjectNavType/val/013653/raw_navigation_camera__0.mp4", "hypernyms": ["instrument.n.01"], "freqs": [15]}'}}

    goals = {0: {'center': (7.5, 5.5), 'radius': 0.4, 'movement':{'type':'static'}}, #goal region for the agentssss
	1: {'center': (1, 1.75), 'radius': 0.4, 'movement':{'type':'static'}}}

    STL_tasks = [
    {"goal_ids": [0, 1], "spec": dict(operator="F", a=0,  b=60,  t_star=50,  gamma_inf=-0.1, collision_penalty=2.0)},
    {"goal_ids": [0, 1], "spec": dict(operator="F", a=80, b=140, t_star=130, gamma_inf=-0.1, collision_penalty=2.0)},
    ]

    STL_horizon = 140 #CHANGE LATER!

    stl_task_str = """
    signal x, y   # signal namesss
    mu_1 := x[t] > 0  # goal-1
    mu_2 := y[t] > 0  # goal-2

    phi1 := F_[0, 60] (mu_1 or mu_2)
    phi2 := F_[80, 140] (mu_1 or mu_2)
    phi := phi1 and phi2
    """


    targets = {}
    #config dictionary for the environment
    config = {
        'house_index': 152,
        'init_loc':[1, 4, 270.0], #initial location of the agent (x, y)
        "dt": 1,
        "render": False,
		'dt_render': 0.01,
		'goals': goals, #goal regions for the agent
        "obstacle_location": [300.0, 300.0],
        "obstacle_size": 0.0,
        "randomize_loc": False, #whether to randomize the agent location at the end of each episode
		'deterministic': False,
		"dynamics": "discrete unicycle", #dynamics model to use
		"targets": targets,
		"disturbance": None, #disturbance range in both x and y directions [w_min, w_max]
		"agent_as_point": False,
        "tasks": STL_tasks,
        "episode_len": STL_horizon,
        "no_reward": True
    }


    #env_2d = Continuous2DEnv(config)

    #Solve HJB:
    house_index = 152

    target_center = (7.0, 5.5)  # Adjust based on your house
    target_radius = 0.5


    # Create dynamics
    #dynamics = house_BRT.Unicycle(max_v=0.2, max_omega=1.0)
    
    # Compute BRT
    # print("Computing BRT...")
    # grid, times, target_values, obstacle_values, all_brt_values, geom = house_BRT.compute_house_brt_over_time(
    #     dynamics=dynamics,
    #     house_index=house_index,
    #     target_center=target_center,
    #     target_radius=target_radius,
    #     time_horizon=20.0,
    #     n_time_steps=21, 
    #     robot_radius=0.2,
    #     wall_thickness=0.1
    # )
    # print("Done!")

    time_horizon = 30.0

    print("Computing BRT in subprocess...")
    brt_data = BRT_subprocess.compute_brt_in_subprocess(
        house_index=152,
        target_center=target_center,
        target_radius=target_radius,
        time_horizon= time_horizon,
        output_path="/tmp/brt_result.pkl"
    )
    print("BRT loaded!")



    num_trials = 1                # number of GOOD episodes you want
    good_episodes = 0               # counts only non-mismatch episodes
    attempts = 0                    # counts all runs (including mismatches)

    num_success_main = 0
    total_eps_len = 0
    num_success_stl = 0
    num_mismatch = 0

    starting_time = time.time()

    while good_episodes < num_trials:
        attempts += 1
        stl_task_done = False  # (make sure you actually set this from results if it's in metrics)

        tasks_queue = mp.Queue()
        results_queue = mp.Queue()

        print(f"Attempt: {attempts} | Good episodes so far: {good_episodes}/{num_trials}")

        tasks_queue.put(task)

        start_worker(
            worker,
            agent_class,
            agent_input,
            device=0,
            tasks_queue=tasks_queue,
            results_queue=results_queue,
        )

        results = results_queue.get()[0]
        metrics = results["metrics"]


        # --- skip mismatch episodes ---
        if metrics.get("mismatch_episode", False):
            num_mismatch += 1
            print("⚠️  Mismatch episode detected -> discarding and rerunning.")
            continue

        # --- count this as a GOOD episode ---
        good_episodes += 1

        success = metrics["success"]
        eps_len = metrics["eps_len"]
        total_eps_len += eps_len

        if stl_task_done:
            num_success_stl += 1

        if success:
            num_success_main += 1



        # ---- Running success rates ----
        main_rate = num_success_main / good_episodes
        stl_rate  = num_success_stl  / good_episodes

        print(
            f"Progress: {good_episodes}/{num_trials} good episodes\n"
            f"  Main task success: {num_success_main}/{good_episodes} "
            f"({main_rate*100:.2f}%)\n"
            f"  STL task success:  {num_success_stl}/{good_episodes} "
            f"({stl_rate*100:.2f}%)"
        )


    finish_time = time.time()
    elapsed_time = finish_time - starting_time
    print("\n===== FINAL RESULTS =====")
    print("Good episodes collected:", good_episodes)
    print("Mismatch episodes discarded:", num_mismatch)
    print("Total attempts:", attempts)
    print("Number of successful main trials:", num_success_main)
    print(f"Main task success rate over {num_trials} GOOD trials: {num_success_main/num_trials*100:.2f}%")
    print(f"Average episode length over {num_trials} GOOD trials: {total_eps_len/num_trials}")
    print("Total time elapsed:", elapsed_time, "seconds")
    print("STL tasks completed in", num_success_stl, "out of", num_trials, "GOOD trials.")

