# Chessboard detection and pieces classification
Computer Vision course final project

## Cloning

Dataset 2 and the pre-trained models (`.yml` files in `output_files/`) are stored with
[Git LFS](https://git-lfs.com/). Install it before cloning.

## Compiling instructions

This code is fully compatible with the Unipd vlab environment (Linux, g++).

After initializing the opencv environment with the command `start_opencv`, navigate to the root 
of the project and execute the following sequence of commands to build the executable.

```bash
mkdir build
cd build
cmake ..
make
```

The `input_files` and `output_files` folders are already in the right place relative to the
project root; no files need to be moved.

The project has been also tested (and developed) in Windows using the Ninja build system and 
the MSVC compiler.

## Dataset 1

Dataset 1 is not included in this repository and must be downloaded from Roboflow:
https://universe.roboflow.com/chduck/merge_chess/dataset/1

Download it in the **YOLOv5 PyTorch** format and extract it into the (empty) `input_files/dataset1/`
folder, so that the files are organized as follows:

```
input_files/dataset1/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## Model training

The repository comes with a pre-trained model in `output_files/`, so training can be skipped.
Dataset 1 is still **required** in any case, because its test split is used for evaluation.