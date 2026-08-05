# Specification-Aware Distribution Shaping for Robotics Foundation Models

Code for enforcing Signal Temporal Logic (STL) specifications on a pretrained robotics foundation model at execution time, without retraining or modifying its parameters.

[![Watch the demo](https://img.youtube.com/vi/ftQ7b_69EnY/hqdefault.jpg)](https://youtu.be/ftQ7b_69EnY)

**Paper:** [arXiv:2603.17969](https://arxiv.org/abs/2603.17969) · **Video:** [YouTube](https://youtu.be/ftQ7b_69EnY)

## Method

Robotics foundation models follow natural language instructions well but give no guarantees about time-dependent requirements — reaching a goal within a deadline, visiting subgoals in order, or keeping a safety condition true throughout an episode.

At each decision step, this framework takes the action distribution produced by the pretrained model and solves a constrained optimization problem: find the minimally modified distribution (in KL divergence) that still satisfies a hard STL feasibility constraint, evaluated by propagating forward dynamics over the remaining horizon. The problem admits a closed-form solution, and satisfaction is guaranteed for deterministic dynamics. The base policy's weights are never touched — the shaping happens entirely at inference.

Experiments use [SPOC](https://github.com/allenai/spoc-robot-training) as the base foundation model, evaluated in AI2-THOR on CHORES-style `ObjectNavType` tasks across procedurally generated houses.

## Repository layout

| Path | Description |
| --- | --- |
| `spoc-robot-training/` | SPOC codebase with the distribution shaping hooks added into the evaluation loop |
| `how_to_run.txt` | Evaluation commands used for the reported experiments |
| `action_results.json` | Logged per-step action distributions from a run |
| `config.yaml` | W&B run config from the pretrained SPOC checkpoint (reference only) |
| `tmp_log/` | Evaluation output directory |

## Setup

Follow the [SPOC installation instructions](https://github.com/allenai/spoc-robot-training) for the environment, AI2-THOR build, and dataset. Then download a pretrained checkpoint — experiments here use `SigLIP-ViTb-3-double-det-CHORES-S` and `SigLIP-ViTb-3-CHORESNav-L`.

## Running

From inside `spoc-robot-training/`:

```bash
python -m training.offline.online_eval \
  --shuffle \
  --eval_subset minival \
  --output_basedir tmp_log \
  --test_augmentation \
  --task_type ObjectNavType \
  --eval_set_size 5 \
  --input_sensors raw_navigation_camera raw_manipulation_camera last_actions an_object_is_in_hand \
  --house_set objaverse \
  --wandb_logging False \
  --num_workers 1 \
  --gpu_devices 0 \
  --training_run_id SigLIP-ViTb-3-CHORESNav-L \
  --local_checkpoint_dir /path/to/pre-trained
```

See `how_to_run.txt` for the double-detection variant, which additionally requires the bounding-box sensors.

## Status

This is research code released for reproducibility, not a maintained library. Paths in `how_to_run.txt` are absolute and need to be changed for your machine, STL specifications are currently defined in code rather than through a config interface, and some logging artifacts are left in the repo. A cleaner interface for specifying STL formulas is planned.

## Citation

```bibtex
@article{yuksel2026specification,
  title   = {Specification-Aware Distribution Shaping for Robotics Foundation Models},
  author  = {Y{\"u}ksel, Sad{\i}k Bera and Aksaray, Derya},
  journal = {arXiv preprint arXiv:2603.17969},
  year    = {2026}
}
```

## Acknowledgments

Built on top of [SPOC](https://github.com/allenai/spoc-robot-training) from the Allen Institute for AI.
