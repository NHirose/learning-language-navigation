import sys
sys.path.insert(0, '/home/vizbot/learning-language-navigation/train')


import matplotlib.pyplot as plt
import os
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

import matplotlib.pyplot as plt
import yaml

# ROS
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32MultiArray
from utils import msg_to_pil, to_numpy, transform_images, load_model, transform_images_lelan

from vint_train.training.train_utils import get_action
import torch
from PIL import Image as PILImage
import numpy as np
import argparse
import yaml
import time
import clip

# UTILS
from topic_names import (IMAGE_TOPIC,
                        WAYPOINT_TOPIC,
                        SAMPLED_ACTIONS_TOPIC,
                        VELOCITY_LELAN_TOPIC)

from transformers import Owlv2Processor, Owlv2ForObjectDetection

# CONSTANTS
TOPOMAP_IMAGES_DIR = "../topomaps/images"
MODEL_WEIGHTS_PATH = "../model_weights"
ROBOT_CONFIG_PATH ="../config/robot.yaml"
MODEL_CONFIG_PATH = "../config/models.yaml"
with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"] 

# GLOBALS
context_queue = []
context_size = None 
obs_img = None 
subgoal = []

# Load the model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def load_owlvit(checkpoint_path="owlv2-base-patch16-ensemble", device='cpu'):
    """
    Return: model, processor (for text inputs)
    """
    processor = Owlv2Processor.from_pretrained(f"google/{checkpoint_path}")
    model = Owlv2ForObjectDetection.from_pretrained(f"google/{checkpoint_path}")
        
    model.to(device)
    model.eval()
    
    return model, processor

def callback_obs(msg):
    global context_size, obs_img
    obs_img = msg_to_pil(msg)
    if context_size is not None:
        if len(context_queue) < context_size + 1:
            context_queue.append(obs_img)
        else:
            context_queue.pop(0)
            context_queue.append(obs_img)

def argmax(lst):
    if not lst:
        raise ValueError("List is empty")
    
    # Initialize the index of the maximum value
    max_index = 0
    
    # Iterate over the list to find the index of the maximum value
    for i in range(1, len(lst)):
        if lst[i] > lst[max_index]:
            max_index = i
    
    return max_index

def find_target_object(topomap, texts):
    model_owl, processor_owl = load_owlvit(checkpoint_path="owlv2-base-patch16-ensemble", device=device)
    size_list = []
    score_list = []
    id_num = 0
    for image in topomap:
        with torch.no_grad():
            inputs_owl = processor_owl(text=texts, images=image, return_tensors="pt").to(device) 
            outputs_owl = model_owl(**inputs_owl)
                
        target_sizes = torch.Tensor([image.size[::-1]])
        results = processor_owl.post_process_object_detection(outputs=outputs_owl, threshold=0.0, target_sizes=target_sizes.to(device))    
    
        i = 0    
        scores = torch.sigmoid(outputs_owl.logits)
        topk_scores, topk_idxs = torch.topk(scores, k=1, dim=1)
        topk_idxs = topk_idxs.squeeze(1).tolist()

        #print(scores, scores.size(), topk_idxs, results[i]['boxes'])
        text = texts[i]    
        topk_boxes = results[i]['boxes'][topk_idxs]
        topk_scores = topk_scores.view(len(text), -1)
        topk_labels = results[i]["labels"][topk_idxs]
        boxes, scores, labels = topk_boxes, topk_scores, topk_labels
        topk_boxes = results[i]['boxes'][topk_idxs]
    
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            size_list.append((box[2]-box[0])*(box[3]-box[1]))
            score_list.append(score.item())
            print("id number", id_num, "score", round(score.item(), 3), "size", (box[2]-box[0])*(box[3]-box[1]))
        
        id_num += 1    
        #im_target = image.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3])))  
    return argmax(score_list), argmax(size_list)

def main(args: argparse.Namespace):
    global context_size, obs_img

     # load model parameters
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)

    model_config_path = model_paths[args.model]["config_path"]
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)

    context_size = model_params["context_size"]

    # load model weights
    ckpth_path = model_paths[args.model]["ckpt_path"]
    if os.path.exists(ckpth_path):
        print(f"Loading model from {ckpth_path}")
    else:
        raise FileNotFoundError(f"Model weights not found at {ckpth_path}")
    model = load_model(
        ckpth_path,
        model_params,
        device,
    )
    model = model.to(device)
    model.eval()

    #loading LeLaN
    ckpth_path_lelan = "../model_weights/wo_col_loss_wo_temp.pth"
    model_config_path_lelan = args.config #"../../train/config/lelan.yaml"

    with open(model_config_path_lelan, "r") as f:
        model_params_lelan = yaml.safe_load(f)

    if os.path.exists(ckpth_path_lelan):
        print(f"Loading LeLaN model from {ckpth_path_lelan}")
    else:
        raise FileNotFoundError(f"Model LeLaN weights not found at {ckpth_path_lelan}")
        
    model_lelan = load_model(
        ckpth_path_lelan,
        model_params_lelan,
        device,
    )
    model_lelan = model_lelan.to(device)
    model_lelan.eval()  
    
     # load topomap
    topomap_filenames = sorted(os.listdir(os.path.join(
        TOPOMAP_IMAGES_DIR, args.dir)), key=lambda x: int(x.split(".")[0]))
    topomap_dir = f"{TOPOMAP_IMAGES_DIR}/{args.dir}"
    num_nodes = len(os.listdir(topomap_dir))
    topomap = []
    for i in range(num_nodes):
        image_path = os.path.join(topomap_dir, topomap_filenames[i])
        topomap.append(PILImage.open(image_path))

    texts = [[args.prompt]]    
                        
    print("Prompts", args.prompt)
    with torch.no_grad():
        batch_obj_inst = clip.tokenize(texts[0][0]).to(device)            
        feat_text = model_lelan("text_encoder", inst_ref=batch_obj_inst)    
    
    nodeid_score, nodeid_size = find_target_object(topomap, texts) 
    print("Target node ID", nodeid_score)
    
    closest_node = 0
    assert -1 <= nodeid_score < len(topomap), "Invalid goal index"
        
    goal_node = nodeid_score
    reached_goal = False

     # ROS
    rospy.init_node("EXPLORATION", anonymous=False)
    rate = rospy.Rate(RATE)
    image_curr_msg = rospy.Subscriber(
        IMAGE_TOPIC, Image, callback_obs, queue_size=1)
    waypoint_pub = rospy.Publisher(
        WAYPOINT_TOPIC, Float32MultiArray, queue_size=1)  
    velocity_lelan_pub = rospy.Publisher(
        VELOCITY_LELAN_TOPIC, Twist, queue_size=1)          
    sampled_actions_pub = rospy.Publisher(SAMPLED_ACTIONS_TOPIC, Float32MultiArray, queue_size=1)
    goal_pub = rospy.Publisher("/topoplan/reached_goal", Bool, queue_size=1)

    print("Registered with master node. Waiting for image observations...")

    if model_params["model_type"] == "nomad":
        num_diffusion_iters = model_params["num_diffusion_iters"]
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=model_params["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
    # navigation loop
    start = 0
    end = goal_node
    while not rospy.is_shutdown():
        # EXPLORATION MODE
        chosen_waypoint = np.zeros(4)
        if len(context_queue) > model_params["context_size"]:
            if model_params["model_type"] == "nomad":
                obs_images = transform_images(context_queue, model_params["image_size"], center_crop=False)
                obs_images = torch.split(obs_images, 3, dim=1)
                obs_images = torch.cat(obs_images, dim=1) 
                obs_images = obs_images.to(device)
                mask = torch.zeros(1).long().to(device)  

                start = max(closest_node - args.radius, 0)
                end = min(closest_node + args.radius + 1, goal_node)
                goal_image = [transform_images(g_img, model_params["image_size"], center_crop=False).to(device) for g_img in topomap[start:end + 1]]
                goal_image = torch.concat(goal_image, dim=0)

                obsgoal_cond = model('vision_encoder', obs_img=obs_images.repeat(len(goal_image), 1, 1, 1), goal_img=goal_image, input_goal_mask=mask.repeat(len(goal_image)))
                dists = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
                dists = to_numpy(dists.flatten())
                min_idx = np.argmin(dists)
                closest_node = min_idx + start
                print("closest node:", closest_node)
                sg_idx = min(min_idx + int(dists[min_idx] < args.close_threshold), len(obsgoal_cond) - 1)
                obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)

                # infer action
                with torch.no_grad():
                    # encoder vision features
                    if len(obs_cond.shape) == 2:
                        obs_cond = obs_cond.repeat(args.num_samples, 1)
                    else:
                        obs_cond = obs_cond.repeat(args.num_samples, 1, 1)
                    
                    # initialize action from Gaussian noise
                    noisy_action = torch.randn(
                        (args.num_samples, model_params["len_traj_pred"], 2), device=device)
                    naction = noisy_action

                    # init scheduler
                    noise_scheduler.set_timesteps(num_diffusion_iters)

                    start_time = time.time()
                    for k in noise_scheduler.timesteps[:]:
                        # predict noise
                        noise_pred = model(
                            'noise_pred_net',
                            sample=naction,
                            timestep=k,
                            global_cond=obs_cond
                        )
                        # inverse diffusion step (remove noise)
                        naction = noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=naction
                        ).prev_sample
                    print("time elapsed:", time.time() - start_time)

                naction = to_numpy(get_action(naction))
                sampled_actions_msg = Float32MultiArray()
                sampled_actions_msg.data = np.concatenate((np.array([0]), naction.flatten()))
                print("published sampled actions")
                sampled_actions_pub.publish(sampled_actions_msg)
                naction = naction[0] 
                chosen_waypoint = naction[args.waypoint]
            elif (len(context_queue) > model_params["context_size"]):
                if not reached_goal:
                    start = max(closest_node - args.radius, 0)
                    end = min(closest_node + args.radius + 1, goal_node)
                    print(start, end)
                    distances = []
                    waypoints = []
                    batch_obs_imgs = []
                    batch_goal_data = []
                    for i, sg_img in enumerate(topomap[start: end + 1]):
                        transf_obs_img = transform_images(context_queue, model_params["image_size"])
                        goal_data = transform_images(sg_img, model_params["image_size"])
                        batch_obs_imgs.append(transf_obs_img)
                        batch_goal_data.append(goal_data)
                    
                    # predict distances and waypoints
                    batch_obs_imgs = torch.cat(batch_obs_imgs, dim=0).to(device)
                    batch_goal_data = torch.cat(batch_goal_data, dim=0).to(device)

                    distances, waypoints = model(batch_obs_imgs, batch_goal_data)
                    distances = to_numpy(distances)
                    waypoints = to_numpy(waypoints)
                    # look for closest node
                    closest_node = np.argmin(distances)
                    # chose subgoal and output waypoints
                    if distances[closest_node] > args.close_threshold:
                        chosen_waypoint = waypoints[closest_node][args.waypoint]
                        sg_img = topomap[start + closest_node]
                    else:
                        chosen_waypoint = waypoints[min(
                            closest_node + 1, len(waypoints) - 1)][args.waypoint]
                        sg_img = topomap[start + min(closest_node + 1, len(waypoints) - 1)]   
                else:
                    im_crop = obs_img.resize((560, 560), PILImage.Resampling.LANCZOS)
                    obs_images = transform_images_lelan(im_crop, [224, 224], center_crop=False)                    
                    obs_images = torch.split(obs_images, 3, dim=1)
                    obs_images = torch.cat(obs_images, dim=1)
                    batch_obs_images = obs_images.to(device)                    
                    with torch.no_grad(): 
                        obsgoal_cond = model_lelan("vision_encoder", obs_img=batch_obs_images, feat_text = feat_text.float())            
                        linear_vel, angular_vel = model_lelan("dist_pred_net", obsgoal_cond=obsgoal_cond)

                    msg_pub = Twist()
                    vt = linear_vel.cpu().numpy()[0,0]
                    wt = angular_vel.cpu().numpy()[0,0]

                    maxv = 0.3
                    maxw = 0.5            

                    if np.absolute(vt) <= maxv:
                        if np.absolute(wt) <= maxw:
                            msg_pub.linear.x = vt
                            msg_pub.angular.z = wt
                        else:
                            rd = vt/wt
                            msg_pub.linear.x = maxw * np.sign(vt) * np.absolute(rd)
                            msg_pub.angular.z = maxw * np.sign(wt)
                    else:
                        if np.absolute(wt) <= 0.001:
                            msg_pub.linear.x = maxv * np.sign(vt)
                            msg_pub.angular.z = 0.0
                        else:
                            rd = vt/wt
                            if np.absolute(rd) >= maxv / maxw:
                                msg_pub.linear.x = maxv * np.sign(vt)
                                msg_pub.angular.z = maxv * np.sign(wt) / np.absolute(rd)
                            else:
                                msg_pub.linear.x = maxw * np.sign(vt) * np.absolute(rd)
                                msg_pub.angular.z = maxw * np.sign(wt)        
                    print(vt, wt)
                    velocity_lelan_pub.publish(msg_pub)
                                                
        # RECOVERY MODE
        if model_params["normalize"]:
            chosen_waypoint[:2] *= (MAX_V / RATE)  
        waypoint_msg = Float32MultiArray()
        waypoint_msg.data = chosen_waypoint
        waypoint_pub.publish(waypoint_msg)
        if model_params["model_type"] != "nomad":
            closest_node += start        
        print("current ID", closest_node)
        if not reached_goal: 
            #reached_goal = closest_node == goal_node
            reached_goal = (closest_node) >= goal_node            
        #reached_goal = True            
            
        goal_pub.publish(reached_goal)
        if reached_goal:
            print("Reached target node! Switch policy to LeLaN...")
        rate.sleep()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to run LeLaN for long-distance navigation with topological map")
    parser.add_argument(
        "--model",
        "-m",
        default="vint",
        type=str,
        help="model name (only nomad is supported) (hint: check ../config/models.yaml) (default: nomad)",
    )
    parser.add_argument(
        "--waypoint",
        "-w",
        default=2, # close waypoints exihibit straight line motion (the middle waypoint is a good default)
        type=int,
        help=f"""index of the waypoint used for navigation (between 0 and 4 or 
        how many waypoints your model predicts) (default: 2)""",
    )
    parser.add_argument(
        "--dir",
        "-d",
        default="topomap",
        type=str,
        help="path to topomap images",
    )
    parser.add_argument(
        "--goal-node",
        "-g",
        default=-1,
        type=int,
        help="""goal node index in the topomap (if -1, then the goal node is 
        the last node in the topomap) (default: -1)""",
    )
    parser.add_argument(
        "--close-threshold",
        "-t",
        default=3,
        type=int,
        help="""temporal distance within the next node in the topomap before 
        localizing to it (default: 3)""",
    )
    parser.add_argument(
        "--radius",
        "-r",
        default=4,
        type=int,
        help="""temporal number of locobal nodes to look at in the topopmap for
        localization (default: 2)""",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        default=8,
        type=int,
        help=f"Number of actions sampled from the exploration model (default: 8)",
    )
    parser.add_argument(
        '--prompt', 
        '-p',
        default = "person",
        type=str, 
        help="prompts of the target objects"
    )
    parser.add_argument(
        '--config', 
        '-c', 
        default = "../../train/config/lelan.yaml", 
        type=str, 
        help="path for the config file (.yaml)"
    )
    args = parser.parse_args()
      
    print(f"Using {device}")
    main(args)


