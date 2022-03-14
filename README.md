<div align="center">
  <h1 align="center">Speedfolding</h1>
  <h3 align="center">
     Learning Efficient Bimanual Folding of Garments
  </h3>
</div>

<p align="center">
 <a href="https://pantor.github.io/speedfolding">
  <img width="360" src="docs/fold-short.gif?raw=true" alt="Folding Video" />
 </a>
 <br>
</p>

This repository contains the code for [Speedfolding: Learning Efficient Bimanual Folding of Garments](https://pantor.github.io/speedfolding), with the corresponding paper submitted to the 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2022). The code is not meant to be easily runnable, as it depends tightly on the hardware setup including the ABB Yumi robot and a Photoneo PhoXi camera. We recommend to see it more as a reference implementation, in particular for our [BiMama-Net]() architecture.


## Installation

The Speedfolding code depends on [abb_librws](https://github.com/mjd3/abb_librws/tree/18f6b42df6b0bb30fb608048911073edb4c71d5b), the [SDK](https://www.photoneo.com/support/) for Photoneo PhoXi camera, as well as on the relevant NVIDIA driver and CUDA installation for PyTorch. Further Python (3.6+) dependencies can then be installed via

```
pip install -r requirements.txt
pip install -e third_party/*
```

The `third_party` components, namely a planner and a camera wrapper, are written in C++ and use pybind11 to expose Python funcionality. For running the scripts, make sure to have both the `learning` and `manipulation` directory in your `PYTHONPATH`.


## Structure

The overall structure is as follows:

- `/data` includes the user-specified instructions for folding garments and calibration data.
- `/database` contains the backend-server for saving data to our database. Have a look at `learning/database.py` for information about reading the database.
- `/learning` is about our introduced BiMama-Net architecture (and everything related to training and running inference with it). It includes scripts for drawing, augmentation, as well as model and reward definitions. `inference.py` is the main class for calculating predictions and the next action.
- `/manipulation` includes everything related to the robot and running the overall high-level pipeline. While `yumi.py` controls the robot itself, `experiment.py` defines the motions for the manipulation primitives. The `heuristics` directory includes all calculations of primitives that are not learned, in particular for instruction matching or folding primitives.


## Entry Points

1. First, start the database by running `uvicorn main:app --app-dir database` from the project root directory. When running the Speedfolding setup, it will upload the most recent images to the database as the *current* image for debugging purposes.
2. To check robot motions or run single manipulation primitives, use `manipulation/experiment.py`. For example, `python manipulation/experiment.py --do-the-fling` will fling the garment from pre-defined pick poses. `python manipulation/experiment.py --instruction shirt` will fold an already smooth shirt according to the defined folding-lines instruction. `python manipulation/experiment.py --do-2s-fold` will apply the *2 seconds* folding heuristic.
3. To run the complete pipeline, use `mainpulation/run.py`. Most of our experiments were run using the `python manipulation/run.py --horizon 10 --demo --fold` arguments. To repeat the experiment even after a robot error, swap `run.py` with `run_forever.py` keeping the same arguments. The overall end-to-end pipeline depends on a number of hyperparameters that are explained furthermore.


## Hyperparameters

| Group        | Parameter                               |  Commonly used value                    |
| ------------ | --------------------------------------- | ---------------------------------------:|
| **Database** | URL                                     | http://127.0.0.1:8000                   |
| **Camera**   | PhoXi Serial number                     | 1703005                                 |
|              | Extrinsic calibration                   | data/calibrations/phoxi_to_world_bww.tf |
| **Motion**   | Speed (Full)                            | 0.6 m/s                                 |
|              | Speed (Half)                            | 0.12 m/s                                |
|              | Speed (Fling)                           | 1.0 m/s                                 |
|              | Speed (Stretch)                         | 0.06 m/s                                |
|              | Force Threshold (Stretch)               | 0.025 N                                 |
|              | Timeout                                 | 7 s                                     |
| **Grasping** | Inwards distance                        | 0.018 m                                 |
|              | Inwards angle                           | 0.25 rad                                |
|              | Gripper force                           | 20 N                                    |
|              | Approach distance                       | 0.04 m                                  |
| **Learning** | Inference image size                    | 256 x 192 px                            |
|              | Depth image distances                   | 0.85 - 1.25 m                           |
|              | Reachability masks                      | data/masks/{left, right}.npy            |
|              | Bimama-Net rotations                    | 20                                      |
|              | Bimama-Net Embedding size               | 8                                       |
|              | Training epochs                         | 100                                     |
|              | Learning rate                           | 4e-4                                    |
|              | Learning rate exponential decay         | 0.97                                    |

These are some of the parameters. In particular, the parametrization of the mainipulation primitives are too complex to state here. We refer to the `manipulation/experiment.py` directly.
