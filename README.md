# LeLaN: Learning A Language-conditioned Navigation Policy from In-the-Wild Video

**Contributors**: Noriaki Hirose<sup>1, 2</sup>, Catherine Glossop<sup>1</sup>\*, Ajay Sridhar<sup>1</sup>\*, Oier Mees<sup>1</sup>, Sergey Levine<sup>1</sup>

<sup>1</sup> UC Berkeley (_Berkeley AI Research_),  <sup>2</sup> Toyota Motor North America, \* indicates equal contributiion


THIS SOFTWARE AND/OR DATA WAS DEPOSITED IN THE BAIR OPEN RESEARCH COMMONS REPOSITORY ON [DATE].

[Project Page](https://learning-language-navigation.github.io) | [Dataset](https://drive.google.com/file/d/1ZwSKwhamq8XmF4mcFNisp9H7YX5kQZ6e/view?usp=sharing) | [Pre-Trained Models](https://drive.google.com/drive/folders/19yJcSJvGmpGlo0X-0owQKrrkPFmPKVt8?usp=sharing)

## Overview
This repository contains code for training our language-conditioned navigation policy with our data, pre-trained model checkpoints, as well as example code to deploy it on a real robot. We made our code by editing the origional code base for training the general navigation models, GNM, ViNT, and NoMaD in this [repository](https://github.com/robodhruv/visualnav-transformer). We try to add our LeLaN code with keeping the original code as much as possible. We appricate the GNM, ViNT, and NoMaD teams (We got their approval to edit and add our codes on their base).

### Preliminary
Please down load our code and install some tools for making a conda environment to run our code. We recommend to run our code in the conda environment.

1. Download the repository on your PC:
    ```
    git clone https://github.com/NHirose/learning-language-navigation.git
    ```
2. Set up the conda environment:
    ```
    conda env create -f train/train_lelan.yml
    ```
3. Source the conda environment:
    ```
    conda activate lelan
    ```
4. Install the lelan packages:
    ```
    pip install -e train/
    ```
5. Install the `diffusion_policy` package from this [repo](https://github.com/real-stanford/diffusion_policy):
    ```
    git clone git@github.com:real-stanford/diffusion_policy.git
    pip install -e diffusion_policy/
    ``` 

### Data
We train our model with the following datasets. We annotate the publicly available robot navigation dataset as well as the in-the-wild videos such as YouTube. In addition, we collect the videos by holding the shperical camera and walking around outside and annotate them by our method. We publish all annotated labels and corresponding images [here](https://drive.google.com/file/d/1ZwSKwhamq8XmF4mcFNisp9H7YX5kQZ6e/view?usp=sharing). Note that we provide the python code to download and save the images from the YouTube videos instead of providing the images, due to avoiding the copyright issue.

- Robot navigation dataset (GO Stanford2, GO Stanford4, and SACSoN)
- Human-walking dataset
- YouTube tour dataset

Followings are the process to use our dataset on our training code.
1. Download the dataset from [here](https://drive.google.com/file/d/1ZwSKwhamq8XmF4mcFNisp9H7YX5kQZ6e/view?usp=sharing) and unzip the file in the downloaded repository:

2. Change the directory:
    ```
    cd learning-language-navigation/download_youtube
    ```
3. Download the YouTube videos and save the corresponding images:
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

2. Camera: Please mount the camera on your robot, which we can use on ROS (publish `sensor_msgs/Image`).

3. Joystick: [Joystick](https://www.amazon.com/Logitech-Wireless-Nano-Receiver-Controller-Vibration/dp/B0041RR0TW)/[keyboard teleop](http://wiki.ros.org/teleop_twist_keyboard) that works with Linux. Add the index mapping for the _deadman_switch_ on the joystick to the `vint_release/deployment/config/joystick.yaml`. You can find the mapping from buttons to indices for common joysticks in the [wiki](https://wiki.ros.org/joy). 


### Software Setup
#### Loading the model weights

Save the model weights *.pth file in `./deployment/model_weights` folder. Our model's weights are in [this link](https://drive.google.com/drive/folders/19yJcSJvGmpGlo0X-0owQKrrkPFmPKVt8?usp=sharing). In addition, we need the original ViNT policy for the long-distance navigation. The ViNT's weights are in [this link](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg).

#### Last-mile Navigation

If the target object location is close to the robot and visible from the robot, you can simply run the LeLaN to move toward the target object. 

1. `roscore`: This launch file opens the `usb_cam` node for the camera, the joy node for the joystick, and nodes for the robot’s mobile base.
2. launch camera node: Please start the camera node to publish the topic, `sensor_msgs/Image`. For example, we use the [usb_cam](http://wiki.ros.org/usb_cam) for the [ELP fisheye camera](https://www.amazon.com/ELP-170degree-Fisheye-640x480-Resolution/dp/B00VTHD17W) and the [cv_camera](http://wiki.ros.org/cv_camera) for the [spherical camera](https://us.ricoh-imaging.com/product/theta-s/). We recommned to use a wide-angle RGB camera to robustly capture the target objects.
3. `python lelan_policy_col.py -p <prompt for target object> -c <path for the config file> -m <path for the moel checkpoint> -r <boolean for camera type>`: This command immediately run the robot toward the target objects, which correspond to the `<prompt for target object>` such as `office chair`. The example of `<path for the config file>` is `'../../train/config/lelan.yaml'` or `'../../train/config/lelan_col.yaml'`, which you used in your training. `<path for the moel checkpoint>` is the path for your trained model. The default is `'../model_weights/wo_col_loss_wo_temp.pth'`. `<bool for camera type>` is the boolean to specify whether the camera is the Ricoh Theta s or not.

Note that you manually change the topic name, 'TOPIC_NAME_CAMERA' in `lelan_policy_col.py`, before running the last command.

#### Long-distance Navigation

If it is difficult for the LeLaN to navigate toward the far target object, you can leverage the topological map to move toward its target object.
There are three steps in our approach, 0) search the all node images and specify the target node capturing the tareget object, 1) move toward the target node, which is close to the target object, and 2) go to the target object location by LeLaN. In our implementation, we use the ViNT policy for 1). To search the target node in the topological memory, we use Owl-ViT2 for scoring all nodes and select the node with the highest score. Before navigation, we collect the topological map in your environment by teleperation. Then we can run our robot toward the far target object.

##### Collecting a Topological Map

_Make sure to run these scripts inside the `./deployment/src/` directory._

##### Record the rosbag: 
Run this command to teleoperate the robot with the joystick and camera. This command opens up three windows 
1. launch the robot driver: please launch the robot driver and setup the node, which eable us to run the robot via a topic of `geometry_msgs/Twist` for the velocity commands, `/cmd_vel`. 
2. launch the camera driver: please launch the `usb_cam` node for the camera. 
3. launch the joystic driver: please launch the joystic driver to publish `/cmd_vel`.
4. `rosbag record /usb_cam/image_raw -o <bag_name>`: This command isn’t run immediately (you have to click Enter). It will be run in the vint_release/deployment/topomaps/bags directory, where we recommend you store your rosbags.

Once you are ready to record the bag, run the `rosbag record` script and teleoperate the robot on the map you want the robot to follow. When you are finished with recording the path, kill the `rosbag record` command, and then kill all sessions.

##### Make the topological map: 
Please open 3 windows and run followings one by one:
1. `roscore`
2. `python create_topomap.py —dt 1 —dir <topomap_dir>`: This command creates a directory in `/learning-language-navigation/deployment/topomaps/images` and saves an image as a node in the map every second the bag is played.
3. `rosbag play -r 1.5 <bag_filename>`: This command plays the rosbag at x1.5 speed, so the python script is actually recording nodes 1.5 seconds apart. The `<bag_filename>` should be the entire bag name with the .bag extension.

When the bag stops playing, kill all sessions.


#### Running the model 
The `<topomap_dir>` is the name of the directory in `vint_release/deployment/topomaps/images` that has the images corresponding to the nodes in the topological map. The images are ordered by name from 0 to N.

Please open 4 windows:

1. launch the robot driver: please launch the robot driver and setup the node, which eable us to run the robot via a topic of `geometry_msgs/Twist` for the velocity commands, `/cmd_vel`. 
2. launch the camera driver: please launch the `usb_cam` node for the camera. 
3. `python pd_controller_lelan.py`: In the graph-based navigation phase, this python script starts a node that reads messages from the `/waypoint` topic (waypoints from the model) and outputs velocities by PD controller to navigate the robot’s base. In the final approach phase, this script selects the velocity commands from the LeLaN.
4. `python navigate_lelan.py -p <prompt> --model vint -—dir <topomap_dir>`: In the graph-based navigation phase, this python script starts a node that reads in image observations from the `/usb_cam/image_raw` topic, inputs the observations and the map into the model, and publishes actions to the `/waypoint` topic. In the final approach phase, this script calculates the LeLaN policy and publishes the velocity commands to the `/vel_lelan` topic.

When the robot is finishing navigating, kill the `pd_controller_lelan.py` script, and then kill all sessions. In the default setting, we run the simplest LeLaN policy not feeding the history of the image and not considering collision avoidance. 

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
