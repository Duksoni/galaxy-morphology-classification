# Galaxy Morphology Classification

### This project is done for course Computer intelligence on Faculty of Technical Sciences, University of Novi Sad.

Technologies used:

![Python](https://img.shields.io/badge/Python-3.12.x-green)
![Tensorflow](https://img.shields.io/badge/Tensorflow-2.20.0-ff7900)
![Keras](https://img.shields.io/badge/Keras-3.11.3-red)

## Prerequisites
- Python Version 3.12.x
- [Galaxy10 DECaLS dataset](https://zenodo.org/records/10845026/files/Galaxy10_DECals.h5) downloaded on your machine
- If you plan to train models on an **NVIDIA** GPU, I highly recommend you to check out this [installation guide](https://www.tensorflow.org/install/pip) to install everything properly for your OS


## Installation

### 1. Download or Clone the Repository

- Download this repository as a zip and extract it to your desired location, or
- Clone the repository:
    ```bash
    git clone https://github.com/Duksoni/galaxy-morphology-classification.git
    ```
    ```bash
    cd galaxy-morphology-classification
    ```

### 2. Place the Galaxy10 dataset .h5 file in the `data` folder in the project root

### 3. (Optional, but recommended) Create a virtual environment:

- On macOS/Linux:

    ```bash
    python3 -m venv .venv
    ```

    ```bash
    source .venv/bin/activate
    ```
- On Windows (if not using WSL or CUDA):
    ```bash
    python -m venv .venv
    ```

    ```bash
    .\.venv\Scripts\activate
    ``` 

### 4. Install Dependencies

- If you want tensorflow/keras to use your **NVIDIA GPU**, run:
    ```bash
    pip install -r requirements-gpu.txt
    ```
- Otherwise, run:
    ```bash
    pip install -r requirements.txt 
    ```


## Running the Application

**Run interactively** — a wrapper script that allows you to configure training parameters for `src/run.py` script 
and allows you to quicly run `src/ui/results.py` script to open the results window.

- On macOS/Linux:
    ```bash
    python3 -m src.quick_run
    ```

- On Windows:
    ```bash
    python -m src.quick_run
    ```

**Run manually** — pass the parameters directly to run.py script, to see all available options add flag `-h` or `--help`.
You must specify model type at minimum.

- On macOS/Linux:
    ```bash
    python3 -m src.run --model_type CNN
    ```

- On Windows:
    ```bash
    python -m src.run --model_type ResNet50 --aug strong_augment
    ```

### Display results
After training, you can run the `src/ui/results.py` script to display the results of the model training and evaluation.

- On macOS/Linux:
    ```bash
    python3 -m src.ui.results
    ```

- On Windows:
    ```bash
    python -m src.ui.results
    ```


