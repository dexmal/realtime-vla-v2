# Realtime-VLA V2: Learning to Run VLAs Fast, Smooth, and Accurate

This repository contains the code for the paper [Realtime-VLA V2: Learning to Run VLAs Fast, Smooth, and Accurate](https://arxiv.org/abs/2603.26360), and provides a deployment stack for real-world dual-arm manipulation with fast, smooth, and accurate execution.

In deployment of VLA models to real-world robotic tasks, execution speed matters. Beyond fast GPU inference, this project focuses on the remaining bottlenecks in the full deployment stack, including calibration, action execution, control, and learning-based speed selection. The end-to-end result is that on real-world tasks requiring both dexterity and accuracy, the robot can execute about `3x` faster than a standard baseline, reaching casual human speed while staying close to the robot hardware limit.

The repository contains:

- `server/`: remote inference service with Pi05 JAX and Pi05 Triton backends, together with time-axis action planning
- `client/`: local runtime stack including robot and camera I/O, observer / actuator bindings, executor implementations, aligned logging, asynchronous video recording, and YAML-based task switching
- modular builder entrypoints in [server/builders.py](server/builders.py) and [client/builders.py](client/builders.py), which make it easy to extend the codebase with custom model backends, robots, observers, actuators, executors, and task configurations

The Triton backend is built on top of [dexmal/realtime-vla](https://github.com/dexmal/realtime-vla) and extends it with realtime chunking / action prefill style usage from [Training-Time Action Conditioning for Efficient Real-Time Chunking](https://arxiv.org/pdf/2512.05964).

## Resources

The table below lists task demos and runtime logs.

| Task | Demo Video | RRD Log |
| --- | --- | --- |
| Cloth Folding | [Demo](https://dexbotic.com/release_video_data/fold_shirt_vla.mp4) | [RRD](https://dexbotic.com/release_video_data/fold_shirt.rrd) |
| Chip Placement | [Demo](https://dexbotic.com/release_video_data/pick_latch_vla.mp4) | [RRD](https://dexbotic.com/release_video_data/pick_latch.rrd) |
| Box Placement | [Demo](https://dexbotic.com/release_video_data/place_into_fixture.mp4) | [RRD](https://dexbotic.com/release_video_data/place_into_fixture.rrd) |

## Installation

The commands below assume you are in the repository root.

```bash
conda create -n realtime-vla-v2 python=3.10 -y
conda activate realtime-vla-v2
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Notes:

- The server is intended to run on an NVIDIA GPU machine compatible with your `torch` and `triton` installation.
- The repository provides a `mock` configuration for running through the end-to-end code path without real robot hardware.
- For real robot deployment, `airbot_real` corresponds to the AIRBOT W1 SDK.
- If you use a different robot stack, you can extend your own robot configuration by adding new implementations and registering them in [client/builders.py](client/builders.py).

## How to Use

All runtime parameters are configured in YAML.

Choose one matching server config and one matching client config for the same task.

### Cloth Folding

Server:

```bash
python server/infer_server.py --config server/config_cloth.yaml
```

Client:

```bash
python client/local_client.py --config client/config_cloth.yaml
```

### Chip Placement

Server:

```bash
python server/infer_server.py --config server/config_chip.yaml
```

Client:

```bash
python client/local_client.py --config client/config_chip.yaml
```

### Box Placement

Server:

```bash
python server/infer_server.py --config server/config_box.yaml
```

Client:

```bash
python client/local_client.py --config client/config_box.yaml
```

### Mock Run

Client:

```bash
python client/local_client.py --config client/config_mock.yaml
```

## Logging

The client saves runtime outputs to the directory specified by `visualization.output_dir` in the selected YAML.

Recording includes:

- aligned trajectory logs in `jsonl`
- asynchronous multi-camera video writing
- in `rrd`, `actual_action` denotes the delay-aligned measured robot state
- for MPC tasks, `raw_pre_mpc_action` denotes the direct model output, `pre_mpc_action` denotes the time-parameterized trajectory before local MPC, and `post_mpc_action` denotes the locally optimized command that is actually sent to the robot
- for smooth / raw-action tasks, `raw_pre_smooth_action` denotes the direct model output, `pre_smooth_action` denotes the time-parameterized trajectory before local smoothing, and `post_smooth_action` denotes the locally smoothed / tracked command that is actually sent to the robot
- inference-complete markers are overlaid on `pre_mpc_action` or `pre_smooth_action` to show inference timing on the trajectory

## Acknowledgements

- [dexmal/realtime-vla](https://github.com/dexmal/realtime-vla)
- [OpenPI](https://github.com/Physical-Intelligence/openpi)
- [Training-Time Action Conditioning for Efficient Real-Time Chunking](https://arxiv.org/pdf/2512.05964)

## Citation
If you want, you can cite this work with:

```bibtex
@article{yang2026realtimevlav2,
  title={Realtime-VLA V2: Learning to Run VLAs Fast, Smooth, and Accurate},
  author={Yang, Chen and Hu, Yucheng and Ma, Yunchao and Yang, Yunhuan and Tan, Jing and Fan, Haoqiang},
  journal={arXiv preprint arXiv:2603.26360},
  year={2026}
}
```
