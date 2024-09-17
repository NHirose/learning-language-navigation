#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '../../train')

#ROS
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

#torch
import torch
import torch.nn.functional as F

#others
import os
import cv2
import yaml
import clip
import argparse
import numpy as np
from PIL import ImageDraw
from PIL import Image as PILImage
from cv_bridge import CvBridge, CvBridgeError
import torchvision.transforms as T

from utils import msg_to_pil, to_numpy, transform_images, transform_images_lelan, load_model, pil2cv, cv2pil

# Create the parser
parser = argparse.ArgumentParser(description="give prompts")
parser.add_argument('-p', '--prompt', type=str, help="prompts of the target objects")
parser.add_argument('-c', '--config', type=str, default = "../../train/config/lelan.yaml", help="path for the config file (.yaml)")
parser.add_argument('-m', '--model', type=str, default = "../model_weights/wo_col_loss_wo_temp.pth", help="path for the config file (.yaml)")
parser.add_argument('-r', '--ricoh', type=bool, default = True, help="True: Ricoh Theta S, False: others")
args = parser.parse_args()

flag_once = 0
store_hist = 0
init_hist = 0
image_hist = []

#Topic name for camera image
TOPIC_NAME_CAMERA = '/cv_camera_node/image_raw'
#TOPIC_NAME_CAMERA = '/usb_cam/image_raw'

# Image Parameters for Ricoh Theta S
xc = 310
yc = 321
yoffset = 310 
xoffset = 310
xyoffset = 280
xplus = 661
XY = [(xc-xyoffset, yc-xyoffset), (xc+xyoffset, yc+xyoffset)]

# load model weights and settings
model_config_path = args.config     # We provide two sample yaml files, "../../train/config/lelan_col.yaml" or "../../train/config/lelan.yaml"
#ckpth_path = "/mnt/sdb/models/wo_col_loss_wo_temp.pth" 
ckpth_path = args.model
# Please down load our checkpoints, with_col_loss.pth, wo_col_loss.pth, and wo_col_loss_wo_temp.pth
# with_col_loss.pth: finetuned model considering collision avoindace loss. We feed the history of image (1 second ago) as well as the current image and the prompt
# wo_col_loss.pth: trained model NOT considering collision avoindace loss. We feed the history of image (1 second ago) as well as the current image and the prompt
# wo_col_loss_wo_temp.pth: trained model NOT considering collision avoindace loss. We feed the current image and the prompt. Simplest model with our core idea.

#ckpth_path = "/mnt/sdb/models/3.pth"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open(model_config_path, "r") as f:
    model_params = yaml.safe_load(f)

if os.path.exists(ckpth_path):
    print(f"Loading model from {ckpth_path}")
else:
    raise FileNotFoundError(f"Model weights not found at {ckpth_path}")

if args.ricoh:
    print("Reading", TOPIC_NAME_CAMERA, "as a spherical camera")
else:
    print("Reading", TOPIC_NAME_CAMERA, "as NOT a spherical camera (canocical or fisheye camera)")
        
model = load_model(
    ckpth_path,
    model_params,
    device,
)
model = model.to(device)
model.eval()  

def preprocess_camera(msg):
    global pub, bridge
    
    if args.ricoh:
        cv2_msg_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        pil_img = cv2pil(cv2_msg_img)
        fg_img = PILImage.new('RGBA', pil_img.size, (0, 0, 0, 255))
        draw=ImageDraw.Draw(fg_img)
        draw.ellipse(XY, fill = (0, 0, 0, 0))
        pil_img.paste(fg_img, (0, 0), fg_img.split()[3])
        cv2_img = pil2cv(pil_img)
        cv_cutimg = cv2_img[yc-xyoffset:yc+xyoffset, xc-xyoffset:xc+xyoffset]
        cv_cutimg = cv2.transpose(cv_cutimg)
        cv_cutimg = cv2.flip(cv_cutimg, 1)
    else:
        cv_cutimg = bridge.imgmsg_to_cv2(msg, "bgr8")
        
    msg_img = bridge.cv2_to_imgmsg(cv_cutimg, 'bgr8')
    pub.publish(msg_img)
        
def callback_lelan(msg_1):
    global init_hist, image_hist
    global flag_once, feat_text

    if True:
        im = msg_to_pil(msg_1)
        #im_crop = im.crop((0, 0, 560, 560))  
        
        # initialize image history
        if init_hist == 0:
            for ih in range(10):
                image_hist.append(im)
            init_hist = 1
                
        im_obs = [image_hist[9], im]
        obs_images, obs_current = transform_images_lelan(im_obs, model_params["image_size"], center_crop=False)              
        obs_images = torch.split(obs_images, 3, dim=1)
        obs_images = torch.cat(obs_images, dim=1)
        batch_obs_images = obs_images.to(device)
        batch_obs_current = obs_current.to(device)
        
        with torch.no_grad():
            # text encoder only once at begging
            if flag_once == 0:
                obj_inst = args.prompt    #"office chair"
                                                                                                                                                                                                                                         
                batch_obj_inst = clip.tokenize(obj_inst).to(device)            
                feat_text = model("text_encoder", inst_ref=batch_obj_inst)
            else:
                flag_once = 1
            
            #LeLaN inference
            if model_params["model_type"] == "lelan_col":
                obsgoal_cond = model("vision_encoder", obs_img=batch_obs_images, feat_text=feat_text.to(torch.float32), current_img=batch_obs_current) 
            elif model_params["model_type"] == "lelan":
                obsgoal_cond = model("vision_encoder", obs_img=batch_obs_current, feat_text=feat_text.to(torch.float32)) 
            linear_vel, angular_vel = model("dist_pred_net", obsgoal_cond=obsgoal_cond)

        
        vt = linear_vel.cpu().numpy()[0,0]
        wt = angular_vel.cpu().numpy()[0,0]
        print("linear vel.", vt, "angular vel.", wt)
        #maximum linear and angular velocity
        maxv = 0.3
        maxw = 0.5           
        msg_pub = Twist()        
        if np.absolute(vt) <= maxv:
            if np.absolute(wt) <= maxw:
                msg_pub.linear.x = vt
                msg_pub.linear.y = 0.0
                msg_pub.linear.z = 0.0
                msg_pub.angular.x = 0.0
                msg_pub.angular.y = 0.0
                msg_pub.angular.z = wt
            else:
                rd = vt/wt
                msg_pub.linear.x = maxw * np.sign(vt) * np.absolute(rd)
                msg_pub.linear.y = 0.0
                msg_pub.linear.z = 0.0
                msg_pub.angular.x = 0.0
                msg_pub.angular.y = 0.0
                msg_pub.angular.z = maxw * np.sign(wt)
        else:
            if np.absolute(wt) <= 0.001:
                msg_pub.linear.x = maxv * np.sign(vt)
                msg_pub.linear.y = 0.0
                msg_pub.linear.z = 0.0
                msg_pub.angular.x = 0.0
                msg_pub.angular.y = 0.0
                msg_pub.angular.z = 0.0
            else:
                rd = vt/wt
                if np.absolute(rd) >= maxv / maxw:
                    msg_pub.linear.x = maxv * np.sign(vt)
                    msg_pub.linear.y = 0.0
                    msg_pub.linear.z = 0.0
                    msg_pub.angular.x = 0.0
                    msg_pub.angular.y = 0.0
                    msg_pub.angular.z = maxv * np.sign(wt) / np.absolute(rd)
                else:
                    msg_pub.linear.x = maxw * np.sign(vt) * np.absolute(rd)
                    msg_pub.linear.y = 0.0
                    msg_pub.linear.z = 0.0
                    msg_pub.angular.x = 0.0
                    msg_pub.angular.y = 0.0
                    msg_pub.angular.z = maxw * np.sign(wt)
        #publish cmd_vel
        pub_vel.publish(msg_pub)

        #update of image history
        image_histx = [im] + image_hist[0:9]
        image_hist = image_histx

def subscriber_callback(msg):
    global latest_image
    latest_image = msg
    
def timer_callback(_):
    global latest_image
    if latest_image is not None:
        callback_lelan(latest_image)
        latest_image = None    

bridge = CvBridge()
latest_image = None

if __name__ == '__main__':    
    #initialize node
    rospy.init_node('LeLaN_col', anonymous=False)

    #subscribe of topics
    rospy.Subscriber('/image_processed', Image, subscriber_callback)
    rospy.Subscriber(TOPIC_NAME_CAMERA, Image, preprocess_camera)
    rospy.Timer(rospy.Duration(0.1), timer_callback)
            
    #publisher of topics
    pub_vel = rospy.Publisher('/cmd_vel', Twist,queue_size=1) #velocities for the robot control
    pub = rospy.Publisher("/image_processed", Image, queue_size = 1)
	
    print('waiting message .....')
    rospy.spin()
