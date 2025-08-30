# Drone Acoustics Hackathon

> The Helsing hackathon challenge for machine learning on drone acoustics.

## Challenge Prompt

Automated detection of threats is essential in facilitating early warning and situational awareness.
Acoustic detection complements other sensor modalities; while radar, optical, and infrared sensors can be
used for drone detection, each has limitations such as weather and obstructions.
Given the low infrastructure costs and ability for rapid deployment, acoustic sensing presents a suitable additional
layer of surveillance for modern defense strategies.

The problem is split into three phases.

### Phase 1: Model Training

We provide a small curated dataset of open-source acoustic recordings split into three categories: background,
drone, and helicopter. The challenge is to train a model to separate these three class from their acoustic signatures.

See "Getting Started" below to begin!

### Phase 2: Competition

Given that you have trained your model in Phase 1, the next step is to compete against other teams at the hackathon!

Go to [http://172.104.137.51:8080/login](http://172.104.137.51:8080/login) to sign up and get instructions on how to
compete.

### Phase 3: Do something awesome!

Now that you've validated your approach by competing against other teams, it's up to you to chose how to extend
this work for the rest of the hackathon!  

If you're looking for inspiration, you could try:
- running your model on an edge device or phone
- using a real microphone and detecting drones in the real world
- using explainability to explore **how** your model identifies drones

## Getting started

To get setup, first clone this repo:
```bash
git clone git@github.com:helsing-ai/edth-copenhagen-drone-acoustics.git
```

This project uses [uv](https://docs.astral.sh/uv/) to manage dependencies; follow the
[uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) if you don't already have it installed.
Then create the virtual environment with:
```bash
uv sync
```

Next, you need to download the train and validation datasets.

> [Train and Validation dataset download](https://github.com/helsing-ai/edth-copenhagen-drone-acoustics/releases/download/train_val_data/drone_acoustics_train_val_data.zip).  

Extract and copy the `train` and `val` directories into `data/raw` so you have `data/raw/train/` and `data/raw/val/`.  
See "Data" below for more details on the dataset used in this work.

You're all ready to go. Work through `intro_notebook.ipynb` for an acoustics machine learning starter guide!

## Data

> [Train and Validation dataset download](https://github.com/helsing-ai/edth-copenhagen-drone-acoustics/releases/download/train_val_data/drone_acoustics_train_val_data.zip).  

See "Getting Started" above for instructions on where to place this data.  
Originally sourced from: https://github.com/DroneDetectionThesis/Drone-detection-dataset (audio + video dataset)  
Paper: [A dataset for multi-sensor drone detection](https://www.sciencedirect.com/science/article/pii/S2352340921007976#!)

### Audio Dataset Details

While the GitHub provides both audio and video, we are only interested in the audio data.  
The challenge is to perform three-class classification (background/drone/helicopter) purely from audio.  
Audio is captured from a Boya BY-MM1 mini cardioid directional microphone with a sampling frequency of 44100 Hz.
We have then applied crop, pitch, volume, and white noise augmentations to the original data to give you more variety.
The background sound class contains general background sounds recorded outdoor in the acquisition location and
includes some clips of the sounds from the servos moving the pan/tilt platform where the sensors were mounted.
