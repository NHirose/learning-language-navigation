# LeLaN: Learning A Language-conditioned Navigation Policy from In-the-Wild Video

**Contributors**: Noriaki Hirose, Catherine Glossop, Ajay Sridhar, Oier Mees, Sergey Levine

_Berkeley AI Research_

[Project Page](https://general-navigation-models.github.io) | [Dataset](https://drive.google.com/file/d/1ZwSKwhamq8XmF4mcFNisp9H7YX5kQZ6e/view?usp=sharing) | [Pre-Trained Models](https://drive.google.com/drive/folders/19yJcSJvGmpGlo0X-0owQKrrkPFmPKVt8?usp=sharing)

## Overview
This repository contains code for training our language-conditioned navigation policy with our data, pre-trained model checkpoints, as well as example code to deploy it on a real robot. We made our code by editing the origional code base for training the general navigation models, GNM, ViNT, and NoMaD [repository](https://github.com/robodhruv/visualnav-transformer). [TODO] indicates the sentences, which we need to edit before releasing. In other words, these sentences are same as the original code base.

- `./train/train.py`: training script to train or fine-tune the LeLaN on your custom data.
- `./train/vint_train/models/`: contains model files for LeLaN and some visual navigation baselines.
- `./deployment/src/record_bag.sh`: script to collect a demo trajectory as a ROS bag in the target environment on the robot. This trajectory is subsampled to generate a topological graph of the environment.
- `./deployment/src/create_topomap.sh`: script to convert a ROS bag of a demo trajectory into a topological graph that the robot can use to navigate.
- `./deployment/src/navigate.sh`: script that deploys a trained GNM/ViNT/NoMaD/LeLaN model on the robot to navigate to a desired goal in the generated topological graph. Please see relevant sections below for configuration settings.

### Data
We train our model with the following datasets. We annotate the publicly available robot navigation dataset as well as the in-the-wild videos such as YouTube. In addition, we collect the videos by holding the shperical camera and walking around and annotate them by our method. We publish all annotated labels and corresponding images [here](https://drive.google.com/file/d/1ZwSKwhamq8XmF4mcFNisp9H7YX5kQZ6e/view?usp=sharing). Note that we provide the python code to download and save the images from the YouTube instead of providing the images, due to the copyright.
[TODO] I need to remove a few trajectories, which is not good for releasing, in Human-walking dataset.

- Robot navigation dataset (GO Stanford2, GO Stanford4, and SACSoN)
- Human-walking dataset
- YouTube tour dataset

Followings are the steps to use our dataset on our training code.
1. Download the repository on your PC:
    ```
    git clone https://github.com/NHirose/learning-language-navigation.git
    ```
2. Download the dataset from [here](https://drive.google.com/file/d/1ZwSKwhamq8XmF4mcFNisp9H7YX5kQZ6e/view?usp=sharing) and unzip the file in the downloaded repository:

3. Change the directory:
    ```
    cd learning-language-navigation/download_youtube
    ```
4. Download the YouTube videos and save the corresponding images:
    ```
    python save_youtube_image.py
    ```
    
## Train

This subfolder contains code for processing datasets and training models from your own data.

### Pre-requisites

The codebase assumes access to a workstation running Ubuntu (tested on 18.04 and 20.04), Python 3.7+, and a GPU with CUDA 10+. It also assumes access to conda, but you can modify it to work with other virtual environment packages, or a native setup.
### Setup
Run the commands below inside the `vint_release/` (topmost) directory:
1. Set up the conda environment:
    ```
    conda env create -f train/train_lelan.yml
    ```
2. Source the conda environment:
    ```
    conda activate lelan
    ```
3. Install the lelan packages:
    ```
    pip install -e train/
    ```
4. Install the `diffusion_policy` package from this [repo](https://github.com/real-stanford/diffusion_policy):
    ```
    git clone git@github.com:real-stanford/diffusion_policy.git
    pip install -e diffusion_policy/
    ```

### Training LeLaN
#### without collision avoidance
Run this inside the `learning-language-navigation/train` directory:
```
python train.py -c ./config/lelan.yaml
```
#### with collision avoidance using the NoMaD supervisions
At first, we need to train the policy without the collision avoidance loss. Then we can finetune it with the collision avoidance loss using the NoMaD supervisions.
Run this inside the `learning-language-navigation/train` directory for pretraining:
```
python train.py -c ./config/lelan_col_pretrain.yaml
```
Then, run this for finetuning (Note that you need to edit the folder name to specify the location of the pretrained model in lelan_col.yaml): 
```
python train.py -c ./config/lelan_col.yaml
```

#### Custom Config Files
`config/lelan.yaml` is the premade yaml files for the LeLaN


#### Training your model from a checkpoint
Instead of training from scratch, you can also load an existing checkpoint from the published results.
Add `load_run: <project_name>/<log_run_name>`to your .yaml config file in `learning-language-navigation/train/config/`. The `*.pth` of the file you are loading to be saved in this file structure and renamed to “latest”: `learning-language-navigation/train/logs/<project_name>/<log_run_name>/latest.pth`. This makes it easy to train from the checkpoint of a previous run since logs are saved this way by default. Note: if you are loading a checkpoint from a previous run, check for the name the run in the `learning-language-navigation/train/logs/<project_name>/`, since the code appends a string of the date to each run_name specified in the config yaml file of the run to avoid duplicate run names. 


If you want to use our checkpoints, you can download the `*.pth` files from [this link](https://drive.google.com/drive/folders/19yJcSJvGmpGlo0X-0owQKrrkPFmPKVt8?usp=sharing).


## Deployment
This subfolder contains code to load a pre-trained LeLaN and deploy it on your robot platform with a [NVIDIA Jetson Orin](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)(We test our policy on Nvidia Jetson Orin AGX). 

### Hardware Setup
We need following three hardwares to navigate the robot toward the target object location with the LeLaN.
1. Robot: Please setup the ROS on your robot to enable us to control the robot by "/cmd_vel" of geometry_msgs/Twist message.

2. Camera: Please mount the camera on your robot and launch the [node](http://wiki.ros.org/usb_cam) to publush "/usb_cam/image_raw" of sensor_msgs/Image message. We recommned to use a wide-angle RGB camera such as [ELP fisheye camera](https://www.amazon.com/ELP-170degree-Fisheye-640x480-Resolution/dp/B00VTHD17W) or [spherical camera](https://us.ricoh-imaging.com/product/theta-s/).

3. Joystick: [Joystick](https://www.amazon.com/Logitech-Wireless-Nano-Receiver-Controller-Vibration/dp/B0041RR0TW)/[keyboard teleop](http://wiki.ros.org/teleop_twist_keyboard) that works with Linux. Add the index mapping for the _deadman_switch_ on the joystick to the `vint_release/deployment/config/joystick.yaml`. You can find the mapping from buttons to indices for common joysticks in the [wiki](https://wiki.ros.org/joy). 


### Software Setup
#### Loading the model weights

Save the model weights *.pth file in `./deployment/model_weights` folder. Our model's weights are in [this link](https://drive.google.com/drive/folders/19yJcSJvGmpGlo0X-0owQKrrkPFmPKVt8?usp=sharing).

#### Last-mile Navigation


#### Long-distance Navigation

##### [TODO]Collecting a Topological Map

_Make sure to run these scripts inside the `vint_release/deployment/src/` directory._


This section discusses a simple way to create a topological map of the target environment for deployment. For simplicity, we will use the robot in “path-following” mode, i.e. given a single trajectory in an environment, the task is to follow the same trajectory to the goal. The environment may have new/dynamic obstacles, lighting variations etc.

##### [TODO]Record the rosbag: 
```bash
./record_bag.sh <bag_name>
```

Run this command to teleoperate the robot with the joystick and camera. This command opens up three windows 
1. `roslaunch vint_locobot.launch`: This launch file opens the `usb_cam` node for the camera, the joy node for the joystick, and nodes for the robot’s mobile base.
2. `python joy_teleop.py`: This python script starts a node that reads inputs from the joy topic and outputs them on topics that teleoperate the robot’s base.
3. `rosbag record /usb_cam/image_raw -o <bag_name>`: This command isn’t run immediately (you have to click Enter). It will be run in the vint_release/deployment/topomaps/bags directory, where we recommend you store your rosbags.

Once you are ready to record the bag, run the `rosbag record` script and teleoperate the robot on the map you want the robot to follow. When you are finished with recording the path, kill the `rosbag record` command, and then kill the tmux session.

##### [TODO]Make the topological map: 
```bash
./create_topomap.sh <topomap_name> <bag_filename>
```

This command opens up 3 windows:
1. `roscore`
2. `python create_topomap.py —dt 1 —dir <topomap_dir>`: This command creates a directory in `/vint_release/deployment/topomaps/images` and saves an image as a node in the map every second the bag is played.
3. `rosbag play -r 1.5 <bag_filename>`: This command plays the rosbag at x5 speed, so the python script is actually recording nodes 1.5 seconds apart. The `<bag_filename>` should be the entire bag name with the .bag extension. You can change this value in the `make_topomap.sh` file. The command does not run until you hit Enter, which you should only do once the python script gives its waiting message. Once you play the bag, move to the screen where the python script is running so you can kill it when the rosbag stops playing.

When the bag stops playing, kill the tmux session.


#### Running the model 
_Make sure to run this script inside the `vint_release/deployment/src/` directory._

```bash
./navigate.sh “--model <model_name> --dir <topomap_dir>”
```

To deploy one of the models from the published results, we are releasing model checkpoints that you can download from [this link](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg?usp=sharing).


The `<model_name>` is the name of the model in the `vint_release/deployment/config/models.yaml` file. In this file, you specify these parameters of the model for each model (defaults used):
- `config_path` (str): path of the *.yaml file in `vint_release/train/config/` used to train the model
- `ckpt_path` (str): path of the *.pth file in `vint_release/deployment/model_weights/`


Make sure these configurations match what you used to train the model. The configurations for the models we provided the weights for are provided in yaml file for your reference.

The `<topomap_dir>` is the name of the directory in `vint_release/deployment/topomaps/images` that has the images corresponding to the nodes in the topological map. The images are ordered by name from 0 to N.

This command opens up 4 windows:

1. `roslaunch vint_locobot.launch`: This launch file opens the usb_cam node for the camera, the joy node for the joystick, and several nodes for the robot’s mobile base).
2. `python navigate.py --model <model_name> -—dir <topomap_dir>`: This python script starts a node that reads in image observations from the `/usb_cam/image_raw` topic, inputs the observations and the map into the model, and publishes actions to the `/waypoint` topic.
3. `python joy_teleop.py`: This python script starts a node that reads inputs from the joy topic and outputs them on topics that teleoperate the robot’s base.
4. `python pd_controller.py`: This python script starts a node that reads messages from the `/waypoint` topic (waypoints from the model) and outputs velocities to navigate the robot’s base.

When the robot is finishing navigating, kill the `pd_controller.py` script, and then kill the tmux session. If you want to take control of the robot while it is navigating, the `joy_teleop.py` script allows you to do so with the joystick.

## Citing
```
@inproceedings{hirose2024lelan,
  title     = {LeLaN: Learning A Language-conditioned Navigation Policy from In-the-Wild Video},
  author    = {Noriaki Hirose and Ajay Sridhar and Catherine Glossop and Oier Mees and Sergey Levine},
  booktitle = {8th Annual Conference on Robot Learning},
  year      = {2024},
  url       = {https://arxiv.org/abs/xxxxxxxx}
}

```
