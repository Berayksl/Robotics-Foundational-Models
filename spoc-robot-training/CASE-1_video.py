#this script is for testing houses for different tasks

import multiprocessing as mp
import os
import platform
import sys
import traceback
from itertools import chain
from queue import Empty as EmptyQueueError
from typing import Literal, Optional, Dict, Any, cast, Sequence, List

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
from environment.stretch_controller_modified import StretchController
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

from robustness_calculator import calculate_goal_predicate_robustness, predicate_square_avoid_rho_over_trajectory

import argparse
from allenact.utils.misc_utils import str2bool
import wandb
import datetime
import time
from online_evaluation.local_logging_utils import LoadLocalWandb, LocalWandb
from architecture.models.transformer_models import REGISTERED_MODELS
import prior
from RL.CASE2.src.simulator_v2 import Continuous2DEnv




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
        house = self.houses[house_idx]
        if house_idx == 9:
            print("Applying house-specific object and window removals for house 9")
            ids_to_remove = ["ObjaFoldingChair|2|2"]
            house = remove_objects_by_id(house, ids_to_remove)
            windows_to_remove = ["window|2|1"]
            house = remove_windows_by_id(house, windows_to_remove)
        return house, house_idx

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
                    "width": 1280,
                    "height": 720,

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

        object_type = task.task_info["synsets"][0]
        object_ids = task.task_info["synset_to_object_ids"][object_type]

        # task_path points out the episode's origin (i.e., which task, episode id, streaming id)
        task_path = "/".join(task.task_info["eval_info"]["task_path"].split("/")[-4:])

        all_frames = []
        all_video_frames = []
        agent.reset()

        # =========================
        # REAL-TIME DISPLAY TOGGLES
        # =========================
        display_realtime = False           # first-person (nav+manip) window
        display_topdown_realtime = True   # topdown third-party camera window
        display_scale = 2                 # scaling for first-person window
        topdown_scale = 1                 # scaling for topdown window (1 = show native res)
        quit_key = ord("q")

        # Optional: also save high-res frames for debugging/figures
        save_topdown_each_step = True
        save_firstperson_hr_each_step = False

        # ======================================================================
        # Add a top-down (map view) third-party camera ONCE per episode
        # ======================================================================
        thor = getattr(task.controller, "controller", task.controller)  # raw ai2thor Controller if wrapped

        ev_cam = thor.step(action="GetMapViewCameraProperties")
        if not ev_cam.metadata.get("lastActionSuccess", False):
            raise RuntimeError(f"GetMapViewCameraProperties failed: {ev_cam.metadata.get('errorMessage')}")

        cam_props = ev_cam.metadata["actionReturn"].copy()
        if "orthographicSize" in cam_props and cam_props["orthographicSize"] is not None:
            cam_props["orthographicSize"] = float(cam_props["orthographicSize"]) #+1 # nicer framing

        ev_add = thor.step(action="AddThirdPartyCamera", skyboxColor="white", **cam_props)
        if not ev_add.metadata.get("lastActionSuccess", False):
            raise RuntimeError(f"AddThirdPartyCamera failed: {ev_add.metadata.get('errorMessage')}")

        # index of the newly added camera
        topdown_cam_idx = -1
        if hasattr(ev_add, "third_party_camera_frames") and ev_add.third_party_camera_frames is not None:
            topdown_cam_idx = len(ev_add.third_party_camera_frames) - 1


        action_list = agent.get_action_list()
        all_actions = []
        additional_metrics = {}
        STL_satisfied = False
        trajectory = []

        actions =  ['r', 'm', 'm', 'm', 'm', 'm', 'm', 'l', 'r', 'm', 'm', 'm', 'm', 'm', 'm', 'l', 'm', 'm', 'b', 'm', 'r', 'm', 'r', 'b', 'r', 'm', 'm', 'm', 'm', 'r', 'm', 'r', 'r', 'r', 'l', 'l', 'm', 'm', 'r', 'l', 'm', 'm', 'm', 'm', 'm', 'r', 'l', 'm', 'm', 'r', 'm', 'r', 'r', 'r', 'r', 'b', 'b', 'r', 'b', 'b', 'm', 'l', 'l', 'r', 'r', 'm', 'm', 'm', 'm', 'm', 'm', 'm', 'm', 'r', 'm', 'm', 'm', 'l', 'm', 'l', 'r', 'l', 'r', 'b', 'l', 'l', 'r', 'r', 'b', 'r', 'l', 'r', 'm', 'l', 'r', 'l', 'r', 'b', 'b', 'l', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'l', 'r', 'r', 'l', 'b', 'b', 'm', 'r', 'm', 'l', 'm', 'm', 'm', 'r', 'm', 'm', 'm', 'm', 'm', 'm', 'm', 'l', 'm', 'm', 'm', 'm', 'm', 'm', 'm', 'm', 'l', 'm', 'end']

        try:
            with torch.no_grad():
                traj_positions = []  # list[dict(x,y,z)] for VisualizePath
                VIS_PATH_EVERY = 1   # set to 5/10 if it’s slow
                for eps_idx in range(task.max_steps):
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

                    # =========================================================
                    # MODEL INPUT (LOW-RES) — keep exactly as expected by SPOC
                    # =========================================================
                    nav_model = task.controller.navigation_camera          # 224x384 (after your crop)
                    manip_model = task.controller.manipulation_camera      # 224x384 (after your crop)
                    #curr_frame = np.concatenate([nav_model, manip_model], axis=1)  # 224x768

                    # =========================================================
                    # HIGH-RES VISUALIZATION FRAMES (do NOT feed to model)
                    # =========================================================
                    # These exist only if you changed controller_args width/height to 1920x1080.
                    nav_hr = task.controller.navigation_camera_hr
                    man_hr = task.controller.manipulation_camera_hr
                    curr_frame = np.concatenate([nav_hr, man_hr], axis=1)  # 224x768

                    # Ensure RGB only
                    if nav_hr is not None and nav_hr.shape[-1] > 3:
                        nav_hr = nav_hr[:, :, :3]
                    if man_hr is not None and man_hr.shape[-1] > 3:
                        man_hr = man_hr[:, :, :3]


                    curr_frame_vis = None
                    if nav_hr is not None and man_hr is not None:
                        # If heights mismatch, resize manip to match nav height
                        if nav_hr.shape[0] != man_hr.shape[0]:
                            man_hr = cv2.resize(
                                man_hr,
                                (man_hr.shape[1], nav_hr.shape[0]),
                                interpolation=cv2.INTER_LINEAR,
                            )

                        # Create a thin vertical separator
                        sep_width = 6  # pixels
                        separator = np.full((nav_hr.shape[0], sep_width, 3), 255, dtype=np.uint8)  # white line

                        # Concatenate: nav | separator | manip
                        curr_frame_vis = np.concatenate([nav_hr, separator, man_hr], axis=1) #RGB

                    # ---------------
                    # Choose action
                    # ---------------
                    action = actions[eps_idx]
                    #action, logits = agent.get_action(observations, goal)
                    #probs = torch.softmax(torch.tensor(logits), -1)

                    # Optional: show first-person in real-time (use HIGH-RES if available)
                    if display_realtime:
                        if curr_frame_vis is not None:
                            h, w = curr_frame_vis.shape[:2]
                            disp = cv2.resize(
                                cv2.cvtColor(curr_frame_vis, cv2.COLOR_RGB2BGR),
                                (int(w * display_scale), int(h * display_scale)),
                                interpolation=cv2.INTER_LINEAR,
                            )
                        else:
                            # Fallback: show model frame if HR not available
                            h, w = curr_frame.shape[:2]
                            disp = cv2.resize(
                                curr_frame,
                                (int(w * display_scale), int(h * display_scale)),
                                interpolation=cv2.INTER_LINEAR,
                            )

                        # cv2.putText(
                        #     disp, f"Task: {goal}", (10, 30),
                        #     cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2
                        # )
                        # cv2.putText(
                        #     disp, f"Time: {eps_idx + 1}", (20, 100),
                        #     cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 8
                        # )
                        # cv2.putText(
                        #     disp, f"Action: {action}", (10, 90),
                        #     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                        # )
                        cv2.imshow("FirstPerson (Nav|Manip) [VIS]", disp)

                    # Keep storing model frames (same behavior as before)
                    #all_frames.append(curr_frame)

                    # Optional: save high-res first-person for figures/debug (RGB->BGR)
                    if save_firstperson_hr_each_step and curr_frame_vis is not None:
                        cv2.imwrite(
                            f"/home/bera/Pictures/SPOC FirstPerson HR/fp_{worker_id}_{eps_idx}.jpg", disp)

                    if self.skip_done and action in ["end", "done"]:
                        action = "sub_done"

                    # Log pose
                    full_pose = task.controller.get_current_agent_full_pose()
                    current_state = (
                        full_pose["position"]["x"],
                        full_pose["position"]["z"],
                        full_pose["rotation"]["y"],
                    )
                    trajectory.append(current_state)

                    # Step environment
                    all_actions.append(action)
                    task.step_with_action_str(action)
                    full_pose = task.controller.get_current_agent_full_pose()
                    x = float(full_pose["position"]["x"])
                    y = float(full_pose["position"]["y"])
                    z = float(full_pose["position"]["z"])

                    traj_positions.append({"x": x, "y": y, "z": z})

                    # =========================================================
                    # Grab TOPDOWN frame from the last event and display realtime
                    # =========================================================
                    thor_last = getattr(task.controller, "controller", task.controller)
                    ev = thor_last.last_event  # most recent event after the action

                    if display_topdown_realtime and hasattr(ev, "third_party_camera_frames"):
                        #frames = ev.third_party_camera_frames
                        ev_path = thor.step(action="VisualizePath", positions=traj_positions)
                        frames = ev_path.third_party_camera_frames
                        if frames is not None and len(frames) > 0:
                            idx = topdown_cam_idx if topdown_cam_idx >= 0 else (len(frames) - 1)
                            topdown_rgb = frames[idx]  # HxWx3 RGB uint8 (at controller res)
                            if topdown_rgb is not None and topdown_rgb.shape[-1] > 3:
                                topdown_rgb = topdown_rgb[:, :, :3]

                            h, w = topdown_rgb.shape[:2]
                            #topdown_rgb = topdown_rgb[:, 360:w-360] #crop to remove the background

                            topdown_bgr_native = cv2.cvtColor(topdown_rgb, cv2.COLOR_RGB2BGR)
                            topdown_disp = topdown_bgr_native


                            scale = 1920 / topdown_disp.shape[1]  # compute automatically

                            h, w = topdown_disp.shape[:2]

                            upscaled = cv2.resize(
                                topdown_disp,
                                (1920, int(h * scale)),
                                interpolation=cv2.INTER_LANCZOS4
                            )

                            h, w = upscaled.shape[:2]
                            upscaled = upscaled[135:h-135, 420:w-420]  #crop to remove the background

                            cv2.putText(
                                upscaled,
                                f"Time:{eps_idx + 1}",
                                (445, 62),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.6,
                                (0, 255, 0),
                                3,
                            )

                            cv2.imshow("TopDown [VIS]", upscaled)



                            # Save: save native-resolution topdown (best quality)
                            if save_topdown_each_step:
                                cv2.imwrite(
                                    f"/home/bera/Pictures/SPOC TopDown Frames/topdown_{worker_id}_{eps_idx}.jpg",
                                    upscaled,
                                )

                    # ---------------
                    # Add bbox overlays (your existing logic) — on MODEL frame
                    # ---------------
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

                    # Logging video uses MODEL frames (expected size)
                    video_frame = self.logging_sensor.get_video_frame(
                        agent_frame=curr_frame,
                        frame_number=eps_idx,
                        action_names=action_list,
                        action_dist=[],
                        ep_length=task.max_steps,
                        last_action_success=task.last_action_success,
                        taken_action=action,
                        task_desc=goal,
                    )
                    #all_video_frames.append(video_frame)

                    # =========================================================
                    # ONE waitKey for BOTH windows (critical for live refresh)
                    # =========================================================
                    if display_realtime or display_topdown_realtime:
                        key = cv2.waitKey(1) & 0xFF
                        if key == quit_key:
                            print("User requested quit")
                            break

                    if task.is_done():
                        print(f"Task is done at step {eps_idx}, breaking out of the loop.")
                        break

        finally:
            if display_realtime or display_topdown_realtime:
                cv2.destroyAllWindows()

        success = task.is_successful()
        print("task success:", success)

        target_ids = None
        if "synset_to_object_ids" in task.task_info:
            target_ids = list(chain.from_iterable(task.task_info.get("synset_to_object_ids", None).values()))

        top_down_frame = get_top_down_frame(
            task.controller, task.task_info["followed_path"], target_ids
        )
        top_down_frame = np.ascontiguousarray(top_down_frame)

        metrics = self.calculate_metrics(task, all_actions, success, additional_metrics)
        metrics["STL_satisfied"] = STL_satisfied

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

def remove_doors_by_asset_id(house_dict, asset_ids_to_remove):
    asset_ids_to_remove = set(asset_ids_to_remove)
    house_dict = dict(house_dict)  # shallow copy
    house_dict["doors"] = [d for d in house_dict.get("doors", [])
                           if d.get("assetId") not in asset_ids_to_remove]
    return house_dict



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

def spawn_transparent_floor_circle(thor_controller, *, x, z, radius, y=0.02, alpha=0.25, rgb=(0.0,1.0,0.0)) -> str:
    ev = thor_controller.step(
        action="CreatePrimitive",
        primitiveType="Cylinder",
        position={"x": float(x), "y": float(y), "z": float(z)},
        rotation={"x": 0.0, "y": 0.0, "z": 0.0},
        # THOR cylinder scale is full size; use radius*2 to get diameter
        scale={"x": float(radius * 2.0), "y": 0.01, "z": float(radius * 2.0)},
    )
    if not ev.metadata.get("lastActionSuccess", False):
        raise RuntimeError(f"CreatePrimitive failed: {ev.metadata.get('errorMessage')}")

    obj_id = ev.metadata["actionReturn"]["objectId"]

    r,g,b = rgb
    ev2 = thor_controller.step(
        action="SetObjectColor",
        objectId=obj_id,
        color={"r": float(r), "g": float(g), "b": float(b), "a": float(alpha)},
    )
    if not ev2.metadata.get("lastActionSuccess", False):
        thor_controller.step(
            action="SetObjectColor",
            objectId=obj_id,
            color={"r": float(r), "g": float(g), "b": float(b)},
        )

    return obj_id

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


    houses[152] = remove_objects_by_id(houses[152], ["FloorLamp|3|1"])
    houses[152] = remove_objects_by_id(houses[152], ["ObjaWheelchair|2|3"])
    houses[152] = remove_objects_by_id(houses[152], ["ObjaTrunk|3|3"])
    houses[152] = remove_objects_by_id(houses[152], ["chair-diningtable-2|2|2|2"])
    houses[152] = remove_objects_by_id(houses[152], ["Bowl|3|30"])
    houses[152] = remove_objects_by_id(houses[152], ['SideTable|2|4'])
    #houses[152] = remove_doors_by_asset_id(houses[152], ["Doorframe_Double_7"])

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

    #find a bowl in house 152:
    task = {'sample_id': 'task=ObjectNavType,house=152,sub_house_id=152', 'house_id': '152', 'task_type': 'ObjectNavType', 'sub_house_id': 152, 'needs_video': False, 'raw_navigation_camera': '', 'sensors_path': '', 
            'observations': {'goal': 'go to a bowl', 'initial_agent_location': np.array([7,  0.90099216,  2 , 270. ,0.]), 'actions': [], 'time_ids': [], 
            'templated_task_type': '{"task_type": "ObjectNavType", "house_index": 152, "agent_starting_position": [0, 0.9009921550750732, 0], "agent_y_rotation": 90.0, "expert_length_bucket": "short", "expert_length": 60, "broad_synset_to_object_ids": {"bowl.n.03": ["Bowl|2|5"]}, "synset_to_object_ids": {"bowl.n.03": ["Bowl|2|5"]}, "synsets": ["bowl.n.03"], "extras": {"chosen_object_id": "Bowl|2|5"}, "natural_language_spec": "go to a bowl", "task_path": "/net/nfs.cirrascale/prior/datasets/vida_datasets/object_nav_v3_benchmark/ObjectNavType/val/013653/raw_navigation_camera__0.mp4", "hypernyms": ["instrument.n.01"], "freqs": [15]}'}}
    
    # #go to a cellphone in house 152:
    # task = {'sample_id': 'task=ObjectNavType,house=152,sub_house_id=152', 'house_id': '152', 'task_type': 'ObjectNavType', 'sub_house_id': 152, 'needs_video': True, 'raw_navigation_camera': '', 'sensors_path': '', 
    #         'observations': {'goal': 'find a cellphone', 'initial_agent_location': np.array([3,  0.90099216,  1 , 270. ,0.]), 'actions': [], 'time_ids': [], 
    #         'templated_task_type': '{"task_type": "ObjectNavType", "house_index": 152, "agent_starting_position": [0, 0.9009921550750732, 0], "agent_y_rotation": 90.0, "expert_length_bucket": "short", "expert_length": 30, "broad_synset_to_object_ids": {"cellular_telephone.n.01": ["CellPhone|3|22"]}, "synset_to_object_ids": {"cellular_telephone.n.01": ["CellPhone|3|22"]}, "synsets": ["cellular_telephone.n.01"], "extras": {"chosen_object_id": "CellPhone|3|22"}, "natural_language_spec": "go to a television", "task_path": "/net/nfs.cirrascale/prior/datasets/vida_datasets/object_nav_v3_benchmark/ObjectNavType/val/013653/raw_navigation_camera__0.mp4", "hypernyms": ["instrument.n.01"], "freqs": [15]}'}}

    # #find a bowl in house 143:
    # task = {'sample_id': 'task=ObjectNavType,house=143,sub_house_id=143', 'house_id': '143', 'task_type': 'ObjectNavType', 'sub_house_id': 143, 'needs_video': False, 'raw_navigation_camera': '', 'sensors_path': '', 
    #         'observations': {'goal': 'find a pan', 'initial_agent_location': np.array([3.5,  0.90099216,  7.5 , 180. ,0.]), 'actions': [], 'time_ids': [], 
    #         'templated_task_type': '{"task_type": "ObjectNavType", "house_index": 143, "agent_starting_position": [0, 0.9009921550750732, 0], "agent_y_rotation": 90.0, "expert_length_bucket": "short", "expert_length": 60, "broad_synset_to_object_ids": {"pan.n.01": ["Pan|2|14"]}, "synset_to_object_ids": {"pan.n.01": ["Pan|2|14"]}, "synsets": ["pan.n.01"], "extras": {"chosen_object_id": "Pan|2|14"}, "natural_language_spec": "find a pan", "task_path": "/net/nfs.cirrascale/prior/datasets/vida_datasets/object_nav_v3_benchmark/ObjectNavType/val/013653/raw_navigation_camera__0.mp4", "hypernyms": ["instrument.n.01"], "freqs": [15]}'}}
    

    # goals = {0: {'center': (7.5, 5.5), 'radius': 0.4, 'movement':{'type':'static'}}, #goal region for the agentssss
	# 1: {'center': (1, 1.75), 'radius': 0.4, 'movement':{'type':'static'}}}
    # obstacles_dict = {}

    # STL_tasks = [
    # {"goal_ids": [0, 1], "spec": dict(operator="F", a=0,  b=60,  t_star=50,  gamma_inf=-0.1, collision_penalty=2.0)},
    # {"goal_ids": [0, 1], "spec": dict(operator="F", a=80, b=140, t_star=130, gamma_inf=-0.1, collision_penalty=2.0)},
    # ]

    # STL_horizon = 140 #CHANGE LATER!

    # stl_task_str = """
    # signal x, y   # signal namesss
    # mu_1 := x[t] > 0  # goal-1
    # mu_2 := y[t] > 0  # goal-2

    # phi1 := F_[0, 60] (mu_1 or mu_2)
    # phi2 := F_[80, 140] (mu_1 or mu_2)
    # phi := phi1 and phi2
    # """


    # targets = {}
    # #config dictionary for the environment
    # config = {
    #     'house_index': 152,
    #     'init_loc':[3.5, 7.5, 180.0], #initial location of the agent (x, y)
    #     "dt": 1,
    #     "render":False,
	# 	'dt_render': 0.01,
	# 	'goals': goals, #goal regions for the agent
    #     'obstacles': obstacles_dict,
    #     "randomize_loc": False, #whether to randomize the agent location at the end of each episode
	# 	'deterministic': False,
	# 	"dynamics": "discrete unicycle", #dynamics model to use
	# 	"targets": targets,
	# 	"disturbance": None, #disturbance range in both x and y directions [w_min, w_max]
	# 	"agent_as_point": False,
    #     "tasks": STL_tasks,
    #     "episode_len": STL_horizon,
    # }

    # env_2d = Continuous2DEnv(config)



    num_trials = 1
    num_success = 0
    total_eps_len = 0
    num_STL_satisfied = 0

    starting_time = time.time()
    for _ in range(num_trials):
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
        STL_done = results["metrics"]["STL_satisfied"]
        
        total_eps_len += eps_len

        if STL_done:
            num_STL_satisfied += 1

        if success == 1.00000001:
            print("The agent successfully completed the task!")
            num_success += 1

    finish_time = time.time()
    elapsed_time = finish_time - starting_time

    print('Number of successful trials:', num_success)
    print('Number of STL satisfied trials:', num_STL_satisfied)
    print(f"Success rate over {num_trials} trials: {num_success/num_trials*100}%")
    print(f"Average episode length over {num_trials} trials: {total_eps_len/num_trials}")
    print('Total time elapsed:', elapsed_time, 'seconds')
    print("STL satisfaction rate over {} trials: {}%".format(num_trials, num_STL_satisfied/num_trials*100))
