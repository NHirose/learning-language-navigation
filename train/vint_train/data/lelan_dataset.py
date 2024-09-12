import numpy as np
import os
import pickle
import yaml
from typing import Any, Dict, List, Optional, Tuple
import tqdm
import io
import lmdb

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

import random
import cv2
import matplotlib.pyplot as plt

from vint_train.data.data_utils import (
    img_path_to_data,
    img_path_to_data_front,
    calculate_sin_cos,
    get_data_path,
    to_local_coords,
) 

class LeLaN_Dataset(Dataset):
    def __init__(
        self,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        waypoint_spacing: int,
        #min_dist_cat: int,
        #max_dist_cat: int,
        #min_action_distance: int,
        #max_action_distance: int,
        #negative_mining: bool,
        len_traj_pred: int,
        learn_angle: bool,
        context_size: int,
        data_split_type: str,
        data_image_folder: str,
        data_pickle_folder: str,   
        #imsize: int,         
        context_type: str = "temporal",
        #end_slack: int = 0,
        #goals_per_obs: int = 1,
        normalize: bool = True,
        obs_type: str = "image",
        goal_type: str = "image",
        backside: bool = False,
        aug_seq: bool = False,
        only_front: bool = False,                     
    ):
        """
        Main ViNT dataset class

        Args:
            data_folder (string): Directory with all the image data
            data_split_folder (string): Directory with filepaths.txt, a list of all trajectory names in the dataset split that are each seperated by a newline
            dataset_name (string): Name of the dataset [recon, go_stanford, scand, tartandrive, etc.]
            waypoint_spacing (int): Spacing between waypoints
            min_dist_cat (int): Minimum distance category to use
            max_dist_cat (int): Maximum distance category to use
            negative_mining (bool): Whether to use negative mining from the ViNG paper (Shah et al.) (https://arxiv.org/abs/2012.09812)
            len_traj_pred (int): Length of trajectory of waypoints to predict if this is an action dataset
            learn_angle (bool): Whether to learn the yaw of the robot at each predicted waypoint if this is an action dataset
            context_size (int): Number of previous observations to use as context
            context_type (str): Whether to use temporal, randomized, or randomized temporal context
            end_slack (int): Number of timesteps to ignore at the end of the trajectory
            goals_per_obs (int): Number of goals to sample per observation
            normalize (bool): Whether to normalize the distances or actions
            goal_type (str): What data type to use for the goal. The only one supported is "image" for now.
            
            backside (str): whether to use the backsie image or not (basically we can use the backside only for Go Stanford 4 and SACSoN.
            aug_seq (str): whether to use the image before and after.
        """
        self.data_split_folder = data_split_folder
        self.data_split_type = data_split_type
        self.data_image_folder = data_image_folder
        self.data_pickle_folder = data_pickle_folder                
        
        self.image_size = image_size
        self.image_size_clip = (224, 224)
        self.waypoint_spacing = waypoint_spacing
        #self.distance_categories = list(
        #    range(min_dist_cat, max_dist_cat + 1, self.waypoint_spacing)
        #)
        #self.min_dist_cat = self.distance_categories[0]
        #self.max_dist_cat = self.distance_categories[-1]
        #self.negative_mining = negative_mining
        #if self.negative_mining:
        #    self.distance_categories.append(-1)
        self.len_traj_pred = len_traj_pred
        self.learn_angle = learn_angle

        #self.min_action_distance = min_action_distance
        #self.max_action_distance = max_action_distance

        self.context_size = context_size
        assert context_type in {
            "temporal",
            "randomized",
            "randomized_temporal",
        }, "context_type must be one of temporal, randomized, randomized_temporal"
        self.context_type = context_type
        #self.end_slack = end_slack
        #self.goals_per_obs = goals_per_obs
        self.normalize = normalize
        self.obs_type = obs_type
        self.goal_type = goal_type

        self.backside = backside
        self.aug_seq = aug_seq
        self.dataset_name = dataset_name
        self.only_front = only_front
        #self.imsize = imsize
        #self.height = height
        #self.width = width
                                
        # load data/data_config.yaml
        with open(
            os.path.join(os.path.dirname(__file__), "data_config.yaml"), "r"
        ) as f:
            all_data_config = yaml.safe_load(f)
        assert (
            self.dataset_name in all_data_config
        ), f"Dataset {self.dataset_name} not found in data_config.yaml"
        dataset_names = list(all_data_config.keys())
        dataset_names.sort()
        # use this index to retrieve the dataset name from the data_config.yaml
        self.dataset_index = dataset_names.index(self.dataset_name)
        self.data_config = all_data_config[self.dataset_name]
        self.trajectory_cache = {}
        #self._load_index()
        #self._build_caches()
        
        self._load_split_index()
        self._build_caches_front()
                
        if self.learn_angle:
            self.num_action_params = 3
        else:
            self.num_action_params = 2

        #norm = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    #def _preprocess_clip(self, image):        

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_image_cache"] = None
        return state
    
    def __setstate__(self, state):
        self.__dict__ = state
        self._build_caches_front()

    def _build_caches_front(self, use_tqdm: bool = True):
        """
        Build a cache of images for faster loading using LMDB
        """
        cache_filename = os.path.join(
            self.data_split_folder,
            f"dataset_{self.dataset_name}_{self.data_split_type}.lmdb",
        )

        # Load all the trajectories into memory. These should already be loaded, but just in case.
        #for traj_name in self.traj_names:
        #    #print("traj_name", traj_name)
        #    self._get_trajectory(traj_name)
        self._get_augdata()
        
        #if self.aug_seq:
        #    self._get_odom()
        
        """
        If the cache file doesn't exist, create it by iterating through the dataset and writing each image to the cache
        """
        if not os.path.exists(cache_filename):
            #print("kiterunokai??")
            #tqdm_iterator = tqdm.tqdm(
            #    self.image_path,
            #    disable=not use_tqdm,
            #    dynamic_ncols=True,
            #    desc=f"Building LMDB cache for {self.dataset_name}"
            #)
            with lmdb.open(cache_filename, map_size=2**40) as image_cache:
                with image_cache.begin(write=True) as txn:
                    print(cache_filename, len(self.image_path))
                    for num in range(len(self.image_path)):
                    #for image_path, time in tqdm_iterator:
                        #print("putting image data into cache", num, len(self.image_path))
                        
                        if os.path.getsize(self.pickle_path[num]) > 0: 
                            with open(self.image_path[num], "rb") as f:
                                txn.put(self.image_path[num].encode(), f.read())
                        else:
                            print(self.image_path[num])

        # Reopen the cache file in read-only mode
        self._image_cache: lmdb.Environment = lmdb.open(cache_filename, readonly=True)

        """
        with self._image_cache.begin() as txn:
            print("_image_cache??? AA", self.image_path[0])
            image_buffer = txn.get(self.image_path[0].encode())
            print("_image_cache??? BB")                
            image_bytes = bytes(image_buffer)
            print("_image_cache??? CC")            
        """
    """
    def _build_index(self, use_tqdm: bool = False):
        samples_index = []
        goals_index = []

        for traj_name in tqdm.tqdm(self.traj_names, disable=not use_tqdm, dynamic_ncols=True):
            traj_data = self._get_trajectory(traj_name)
            traj_len = len(traj_data["position"])

            for goal_time in range(0, traj_len):
                goals_index.append((traj_name, goal_time))

            begin_time = self.context_size * self.waypoint_spacing
            end_time = traj_len - self.end_slack - self.len_traj_pred * self.waypoint_spacing
            for curr_time in range(begin_time, end_time):
                max_goal_distance = min(self.max_dist_cat * self.waypoint_spacing, traj_len - curr_time - 1)
                samples_index.append((traj_name, curr_time, max_goal_distance))

        return samples_index, goals_index
    """
    """
    def _sample_goal(self, trajectory_name, curr_time, max_goal_dist):
        goal_offset = np.random.randint(0, max_goal_dist + 1)
        if goal_offset == 0:
            trajectory_name, goal_time = self._sample_negative()
            return trajectory_name, goal_time, True
        else:
            goal_time = curr_time + int(goal_offset * self.waypoint_spacing)
            return trajectory_name, goal_time, False
    """
    def _sample_negative(self):
        """
        Sample a goal from a (likely) different trajectory.
        """
        return self.goals_index[np.random.randint(0, len(self.goals_index))]
    
    """
    def _load_index(self) -> None:
        index_to_data_path = os.path.join(
            self.data_split_folder,
            f"dataset_dist_{self.min_dist_cat}_to_{self.max_dist_cat}_context_{self.context_type}_n{self.context_size}_slack_{self.end_slack}.pkl",
        )
    """
        
    def _remove_values_from_list(self, A, B):
        return [item for item in A if item not in B]
            
    def _load_split_index(self):
        if self.dataset_name == "go_stanford4":
            """
            split = 10        
            if self.data_split_type == "train":
                index = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            else:
                index = [9]
            """
            self.v_random = 0.2 #for random cropping
            self.h_random = 0.1 #for random cropping
            
            lst = os.listdir(self.data_image_folder) # your directory path
            number_files = len(lst)
            
            image_path = []
            pickle_path = []
            #id_num = []   
            
            ratio = 0.9
            thres = int(number_files*ratio)
            
            if self.data_split_type == "train":
                print("go_stanford4 train flame num", thres)
            else:
                print("go_stanford4 test flame num", number_files-thres)            
             
            #TODO -5 is come from "self.data_image_folder" includes 5 files, which is not pickle file.
            for num in range(int(number_files - 5)-3):
                if self.data_split_type == "train" and num < thres:
                    image_path.append(self.data_image_folder + str(num).zfill(8) + '.jpg')
                    pickle_path.append(self.data_pickle_folder + str(num).zfill(8) + '.pkl')                        
                    #id_num.append(num)            
                elif self.data_split_type == "test" and num >= thres:
                    image_path.append(self.data_image_folder + str(num).zfill(8) + '.jpg')
                    pickle_path.append(self.data_pickle_folder + str(num).zfill(8) + '.pkl')                        
                    #id_num.append(num)                  
            """        
            for cid in index:
                for num in range(int((number_files-1)/(9*split))):
                    #print(num, cid, split, 9*split*num+1+9*cid)
                    image_path.append(self.data_image_folder + 'img_full_' + str(9*split*num+1+9*cid) + '.jpg')
                    pickle_path.append(self.data_pickle_folder + 'data_aug_' + str(9*split*num+1+9*cid) + '.pkl')                        
                    id_num.append(9*split*num+1+9*cid)
            """        
        if self.dataset_name == "sacson":
            """
            split = 10        
            if self.data_split_type == "train":
                index = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            else:
                index = [9]
            """
            self.v_random = 0.2 #for random cropping
            self.h_random = 0.1 #for random cropping 
                        
            image_path = []
            pickle_path = []
            #id_num = []     
                                     
            folder_lst = next(os.walk(self.data_pickle_folder))[1]

            if self.data_split_type == "train":
                folder_lst_dataset = folder_lst[0:len(folder_lst)-1]
                print("SACSoN dataset train seq. number", len(folder_lst_dataset))
            else:
                folder_lst_dataset = folder_lst[len(folder_lst)-1:len(folder_lst)]
                print("SACSoN dataset test seq. number", len(folder_lst_dataset))
                                
            for folder in folder_lst_dataset:
                print(self.data_split_type, folder)
                subfolder_lst = os.listdir(self.data_pickle_folder + folder + "/")                                    
                for subfolder in subfolder_lst:
                    file_lst = os.listdir(self.data_image_folder + folder + "/" + subfolder + "/image/")
                    number_files = len(file_lst)
                    for num in range(int(number_files)-3):
                        image_path.append(self.data_image_folder + folder + "/" + subfolder + "/image/" + str(num).zfill(8) + '.jpg')
                        pickle_path.append(self.data_pickle_folder + folder + "/" + subfolder + "/pickle/" + str(num).zfill(8) + '.pkl')                        
                        #id_num.append(num)                                                                                                
            """
            for folder in folder_lst:
                print(folder)
                subfolder_lst = os.listdir(self.data_pickle_folder + folder + "/")
                for subfolder in subfolder_lst:
                    file_lst = os.listdir(self.data_image_folder + folder + "/" + subfolder + "/360img/")
                    number_files = len(file_lst)
                    for cid in index:
                        for num in range(int((number_files)/(27*split))):
                            #print(num, cid, split, 9*split*num+1+9*cid)
                            image_path.append(self.data_image_folder + folder + "/" + subfolder + "/360img/" + str(27*split*num + 27*cid).zfill(8) + '.jpg')
                            pickle_path.append(self.data_pickle_folder + folder + "/" + subfolder + "/" + str(27*split*num + 27*cid).zfill(8) + '.pkl')                        
                            id_num.append(27*split*num + 27*cid) 
            """
        if self.dataset_name == "go_stanford2":
            """
            split = 10        
            if self.data_split_type == "train":
                index = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            else:
                index = [9]
            """
            self.v_random = 0.2 #for random cropping
            self.h_random = 0.1 #for random cropping 
                        
            image_path = []
            pickle_path = []
            #id_num = []     
                                     
            folder_lst = next(os.walk(self.data_pickle_folder))[1]
            num_test = int(0.1*len(folder_lst))
            
            if self.data_split_type == "train":
                folder_lst_dataset = folder_lst[0:len(folder_lst)-num_test]
                print("go_stanford2 train seq. number", len(folder_lst_dataset))
            else:
                folder_lst_dataset = folder_lst[len(folder_lst)-num_test:len(folder_lst)]
                print("go_stanford2 test seq. number", len(folder_lst_dataset))
            
            for folder in folder_lst_dataset:
                #print("go_stanford2", self.data_split_type, self.data_pickle_folder, folder)
                #subfolder_lst = os.listdir(self.data_pickle_folder + folder + "/")
                #for subfolder in subfolder_lst:
                file_lst = os.listdir(self.data_image_folder + folder + "/image/")
                number_files = len(file_lst)
                for num in range(int(number_files-3)):
                    #print(num, cid, split, 9*split*num+9*cid)
                    image_path.append(self.data_image_folder + folder + "/image/" + str(num).zfill(8) + '.jpg')
                    pickle_path.append(self.data_pickle_folder + folder + "/pickle/" + str(num).zfill(8) + '.pkl')                        
                    #id_num.append(num)        

        if self.dataset_name == "humanw":
            self.v_random = 0.2 #for random cropping
            self.h_random = 0.1 #for random cropping 
                    
            image_path = []
            pickle_path = []
            #id_num = []     
                                     
            folder_lst = next(os.walk(self.data_pickle_folder))[1]
            #num_test = int(0.1*len(folder_lst))
            
            test_folder = ["R0010096", "R0010098","R0010121", "R0010118","R0010133", "R0010145", "R0010156", "R0010166", "R0010175","R0010180", "R0010188", "R0010197"]
            
            if self.data_split_type == "train":
                folder_lst_dataset = self._remove_values_from_list(folder_lst, test_folder)
                print("Human-walking dataset train seq. number", len(folder_lst_dataset))
            else:
                folder_lst_dataset = test_folder
                print("Human-walking dataset test seq. number", len(folder_lst_dataset))
            
            for folder in folder_lst_dataset:
                #print("Human-walking", self.data_split_type, self.data_pickle_folder, folder)
                #subfolder_lst = os.listdir(self.data_pickle_folder + folder + "/")
                #for subfolder in subfolder_lst:
                file_lst = os.listdir(self.data_image_folder + folder + "/image/")
                number_files = len(file_lst)
                for num in range(int(number_files)):
                    #print(num, cid, split, 9*split*num+9*cid)
                    image_path.append(self.data_image_folder + folder + "/image/" + str(num).zfill(8) + '.jpg')
                    pickle_path.append(self.data_pickle_folder + folder + "/pickle/" + str(num).zfill(8) + '.pkl')                        
                    #id_num.append(num)   

        if self.dataset_name == "youtube":
            self.v_random = 0.05 #for random cropping
            self.h_random = 0.05 #for random cropping 
                    
            image_path = []
            pickle_path = []
            #id_num = []     
                                     
            folder_lst = next(os.walk(self.data_pickle_folder))[1]
            #num_test = int(0.1*len(folder_lst))

            test_folder = ["home_10", "home_12", "austra_1", "spain_1", "singa_1", "spain_3", "spain_5", "rosia_2", "home_33", "poland_1", "uk_5"]
              
            if self.data_split_type == "train":
                folder_lst_dataset = self._remove_values_from_list(folder_lst, test_folder)
                print("YouTube dataset train seq. number", len(folder_lst_dataset))                
            else:
                folder_lst_dataset = test_folder
                print("YouTube dataset test seq. number", len(folder_lst_dataset))   
            
            for folder in folder_lst_dataset:
                #print("YouTube", self.data_split_type, self.data_pickle_folder, folder)
                #subfolder_lst = os.listdir(self.data_pickle_folder + folder + "/")
                #for subfolder in subfolder_lst:
                file_lst = os.listdir(self.data_image_folder + folder + "/image/")
                number_files = len(file_lst)
                for num in range(int(number_files)):
                    #print(num, cid, split, 9*split*num+9*cid)
                    image_path.append(self.data_image_folder + folder + "/image/" + str(num).zfill(8) + '.jpg')
                    pickle_path.append(self.data_pickle_folder + folder + "/pickle/" + str(num).zfill(8) + '.pkl')                        
                    #id_num.append(num)  
          
        self.image_path = image_path
        self.pickle_path = pickle_path
        
    def _load_image(self, trajectory_name, time):
        image_path = get_data_path(self.data_folder, trajectory_name, time)
        #print("path", image_path, image_path.encode())

        try:
            with self._image_cache.begin() as txn:
                image_buffer = txn.get(image_path.encode())
                image_bytes = bytes(image_buffer)
            image_bytes = io.BytesIO(image_bytes)
            return img_path_to_data(image_bytes, self.image_size)
        except TypeError:
            print(f"Failed to load image {image_path}")

    def _load_image_front(self, path):
        image_path = path

        #with self._image_cache.begin() as txn:
        #    print("_image_cache??? AA")
        #    image_buffer = txn.get(image_path.encode())
        #    print("_image_cache??? BB")                
        #    image_bytes = bytes(image_buffer)
        
        #print(image_path)
        try:
            with self._image_cache.begin() as txn:
                #print("_image_cache??? A", image_path.encode())
                image_buffer = txn.get(image_path.encode())
                #print("image_buffer", image_buffer.max, image_buffer.min)
                #print("_image_cache??? B")                
                image_bytes = bytes(image_buffer)
                #print("_image_cache??? C")                
            image_bytes = io.BytesIO(image_bytes)
            #print("image_bytes", image_bytes.max, image_bytes.min)
            #print("_image_cache??? D")            
            return img_path_to_data_front(image_bytes, self.image_size)
        except TypeError:
            print(f"Failed to load image {image_path}")

    def _compute_actions(self, traj_data, curr_time, goal_time):
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred * self.waypoint_spacing + 1
        yaw = traj_data["yaw"][start_index:end_index:self.waypoint_spacing]
        positions = traj_data["position"][start_index:end_index:self.waypoint_spacing]
        goal_pos = traj_data["position"][min(goal_time, len(traj_data["position"]) - 1)]

        if len(yaw.shape) == 2:
            yaw = yaw.squeeze(1)

        if yaw.shape != (self.len_traj_pred + 1,):
            const_len = self.len_traj_pred + 1 - yaw.shape[0]
            yaw = np.concatenate([yaw, np.repeat(yaw[-1], const_len)])
            positions = np.concatenate([positions, np.repeat(positions[-1][None], const_len, axis=0)], axis=0)

        assert yaw.shape == (self.len_traj_pred + 1,), f"{yaw.shape} and {(self.len_traj_pred + 1,)} should be equal"
        assert positions.shape == (self.len_traj_pred + 1, 2), f"{positions.shape} and {(self.len_traj_pred + 1, 2)} should be equal"

        waypoints = to_local_coords(positions, positions[0], yaw[0])
        goal_pos = to_local_coords(goal_pos, positions[0], yaw[0])

        assert waypoints.shape == (self.len_traj_pred + 1, 2), f"{waypoints.shape} and {(self.len_traj_pred + 1, 2)} should be equal"

        if self.learn_angle:
            yaw = yaw[1:] - yaw[0]
            actions = np.concatenate([waypoints[1:], yaw[:, None]], axis=-1)
        else:
            actions = waypoints[1:]
        
        if self.normalize:
            actions[:, :2] /= self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing
            goal_pos /= self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing

        assert actions.shape == (self.len_traj_pred, self.num_action_params), f"{actions.shape} and {(self.len_traj_pred, self.num_action_params)} should be equal"

        return actions, goal_pos
    
    def _get_trajectory(self, trajectory_name):
        if trajectory_name in self.trajectory_cache:
            return self.trajectory_cache[trajectory_name]
        else:
            with open(os.path.join(self.data_folder, trajectory_name, "traj_data.pkl"), "rb") as f:
                traj_data = pickle.load(f)
                #print(trajectory_name, traj_data)
            self.trajectory_cache[trajectory_name] = traj_data
            return traj_data

    def _resize_norm(self, image, size):
        #return TF.resize(image, size)
        return TF.resize(2.0*(image - 0.5), size)
        #return TF.resize((image - 127.5)/127.5, size)

    def _vis_dataset(self, image_torch, data_aug):
        fig = plt.figure(figsize=(34, 16), dpi=80)
        gs = fig.add_gridspec(20,6)
        ax_img = fig.add_subplot(gs[0:4, 0:2])
        ax_graph = fig.add_subplot(gs[4:20, 0:2])
        ax_crop = []
        ax_text = []
        ax_num = []
        ax_text2 = []    
        black_list = [["modern,", "sleek,", "and", "minimalist"], ["person,", "possibly", "a", "woman,", "wearing", "a", "dark-colored", "outfit."]]        
        
        img_cv2_bgr = torch.permute(torch.flip(image_torch, dims=[0]), (1, 2, 0)).cpu().numpy()
        for ia in range(20):
            ax_num.append(fig.add_subplot(gs[ia, 3]))
            ax_crop.append(fig.add_subplot(gs[ia, 2]))
            ax_text.append(fig.add_subplot(gs[ia, 4]))
            ax_text2.append(fig.add_subplot(gs[ia, 5]))
                 
        for nob in range(len(data_aug)):
            [y_min, y_max, x_min, x_max] = data_aug[nob]["bbox"]
            img_cv2_rgb = cv2.cvtColor(img_cv2_bgr[y_min:y_max, x_min:x_max], cv2.COLOR_BGR2RGB)    
    
            prompt = None
            prompt2 = None
            if nob < 20:
                ax_crop[nob].imshow(img_cv2_rgb)
                ax_crop[nob].axis('off')

                if data_aug[nob]["obj_detect"]:
                    flag_p = 0
                    print("prompt_2" in data_aug[nob], data_aug[nob])
                    if "prompt_2" in data_aug[nob]:
                        prompt = data_aug[nob]["prompt_2"][0]
                        flag_p = 1
                        for ip in range(len(data_aug[nob]["prompt_2"]) - 1):         
                            if ip < 3:               
                                prompt = prompt + "\n" + data_aug[nob]["prompt_2"][ip + 1]
                            elif ip == 3:
                                prompt2 = data_aug[nob]["prompt_2"][ip + 1]
                            elif ip < 7:
                                prompt2 = prompt2 + "\n" + data_aug[nob]["prompt_2"][ip + 1]             
 
                        ax_text[nob].text(-0.5, 0, prompt, fontsize = 8, color = 'red')
                        ax_text2[nob].text(-0.5, 0, prompt2, fontsize = 8, color = 'red')                                         
                    else:
                        print("prompt2 end!", nob, flag_p)
                    
                    if flag_p == 0:    
                        prompt = data_aug[nob]["prompt_1"][0]
                        for ip in range(len(data_aug[nob]["prompt_1"]) - 1):
                            if ip < 3:               
                                prompt = prompt + "\n" + data_aug[nob]["prompt_1"][ip + 1]
                            elif ip == 3:
                                prompt2 = data_aug[nob]["prompt_1"][ip + 1]
                            elif ip < 7:
                                prompt2 = prompt2 + "\n" + data_aug[nob]["prompt_1"][ip + 1] 
                        print("prompt1 end!", nob, flag_p)                         
                        ax_text[nob].text(-0.5, 0, prompt, fontsize = 8)
                        ax_text2[nob].text(-0.5, 0, prompt2, fontsize = 8)   

                    if nob == 14:
                        print("prompt", prompt)
                        print("prompt2", prompt2)
                                                         
                else:
                    ax_text[nob].text(-0.5, 0, "No identified object", fontsize = 8, color="blue")
                
                obj_inst_list = data_aug[nob]["obj_inst"].split(" ") 
                flag_remove = 0
                for l in range(len(black_list)):
                    for bc in range(len(obj_inst_list)-len(black_list[l])+1):
                        #if nob == 11:
                        #    print(obj_inst_list[bc:bc + len(black_list[l])])
                        #    print(black_list[l])                
                        if obj_inst_list[bc:bc + len(black_list[l])] == black_list[l]:
                            flag_remove = 1
            
                flag_largeness = 0

                if (y_max - y_min)*(x_max - x_min) > int(0.5*580*580):
                    flag_largeness = 1
            
                obj_inst_show = obj_inst_list[0]
                for io in range(len(obj_inst_list)-1):
                    if io < 11:               
                        obj_inst_show = obj_inst_show + " " + obj_inst_list[io + 1]
                    elif io == 11:
                        obj_inst_show = obj_inst_show + "\n" + obj_inst_list[io + 1]                    
                    elif io < 22 and io > 11:
                        obj_inst_show = obj_inst_show + " " + obj_inst_list[io + 1] 
                    elif io == 22:               
                        obj_inst_show = obj_inst_show + "\n" + obj_inst_list[io + 1]  
                    elif io < 33 and io > 22:
                        obj_inst_show = obj_inst_show + " " + obj_inst_list[io + 1] 
                    elif io == 33:               
                        obj_inst_show = obj_inst_show + "\n" + obj_inst_list[io + 1]  
                    elif io < 44 and io > 33:
                        obj_inst_show = obj_inst_show + " " + obj_inst_list[io + 1] 
                                                    
                ax_text[nob].axis('off')
                ax_text2[nob].axis('off')               
            
                if flag_remove == 1:
                    ax_num[nob].text(-0.5, 0, str(nob) + ":" + obj_inst_show, fontsize = 8, color = 'green')
                elif flag_largeness == 1:
                    ax_num[nob].text(-0.5, 0, str(nob) + ":" + obj_inst_show, fontsize = 8, color = 'magenta')                      
                else:
                    ax_num[nob].text(-0.5, 0, str(nob) + ":" + obj_inst_show, fontsize = 8, color = 'black')              
                ax_num[nob].axis('off')
                
                [xob, yob, zob] = data_aug[nob]["pose_median"]
                ax_graph.plot(xob, zob,'r*',markersize=10)
                sc = 1.2 + 0.5*random.random()
                ax_graph.plot([xob, sc*xob], [zob, sc*zob], color='red')
                ax_graph.text((sc+0.15)*xob, (sc+0.15)*zob, str(nob), fontsize = 15, color='red')  
        
        for n in range(nob, 20, 1):
            ax_crop[n].axis('off')
            ax_text[n].axis('off')
            ax_text2[n].axis('off')        
            ax_num[n].axis('off')
                
        ax_img.imshow(cv2.cvtColor(img_cv2_bgr, cv2.COLOR_BGR2RGB))
        ax_img.axis('off')
        plt.show()


    def _get_augdata(self, ):
        aug_data_list = []
        #print("len(self.pickle_path)", len(self.pickle_path))
        for num in range(len(self.pickle_path)):
            #print("putting pickle data into cache", num, len(self.pickle_path))
            
            if os.path.getsize(self.pickle_path[num]) > 0:            
                with open(self.pickle_path[num], "rb") as f:
                    aug_data = pickle.load(f)
            else:
                print(self.pickle_path[num])
            aug_data_list.append(aug_data)
            
        self.aug_data_list = aug_data_list    

    #def _get_odom(self, ):
    #    with open(self.odom_text) as f:
    #        lines_odom = f.readlines() 
    #        
    #    odom_list = []
    #    for j in range(len(lines_odom)):         
    #        pose_list_s = [float(i) for i in lines_odom[j].split()]
    #        odom_list.append(pose_list_s)
    #    self.odom_list = odom_list  

    def __len__(self) -> int:
        return len(self.image_path)

    #def __len__(self) -> int:
    #    return len(self.index_to_data)    

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        """
        Args:
            i (int): index to ith datapoint
        Returns:
            Tuple of tensors containing the context, observation, goal, transformed context, transformed observation, transformed goal, distance label, and action label
                obs_image (torch.Tensor): tensor of shape [3, H, W] containing the image of the robot's observation
                goal_image (torch.Tensor): tensor of shape [3, H, W] containing the subgoal image 
                dist_label (torch.Tensor): tensor of shape (1,) containing the distance labels from the observation to the goal
                action_label (torch.Tensor): tensor of shape (5, 2) or (5, 4) (if training with angle) containing the action labels from the observation to the goal
                which_dataset (torch.Tensor): index of the datapoint in the dataset [for identifying the dataset for visualization when using multiple datasets]
        """
        #goal_image_test = self._load_image_front("/home/noriaki/visualnav-transformer/gnm_dataset/go_stanford/sim11hF_4/24.jpg") #self.image_path[0]
        
        flag_data = 0
        flag_data_inner = 0
        iv = i
        
        ib = random.random()
        flag_back = 0
        
        if self.backside and ib > 0.5:
            flag_back = 1
        
        #remove data without object in front of the robot
        while flag_data == 0:
            #print(self.image_path[iv]) #/media/noriaki/Noriaki_Data2/data_auto_collection/Feb-03-2023/00000005/360img
            #if self.dataset_name == "sacson":
            #    image_fullsize = self._load_image_front("/media/noriaki/Noriaki_Data2/data_auto_collection/Feb-14-2023/00000013/360img/00000954.jpg")
            #else:
            #    image_fullsize = self._load_image_front(self.image_path[iv])
            image_fullsize = self._load_image_front(self.image_path[iv]) #self.image_path[0]
            flag_data_inner = 0
            
            context_image = [image_fullsize]        
            #print("context_size", self.context_size)    
            for ih in range(self.context_size):
                if iv-ih > 0:                   
                    context_image.append(self._load_image_front(self.image_path[iv-ih]))             
                else:
                    context_image.append(self._load_image_front(self.image_path[0]))
            #print("context_image size", len(context_image))
            
            for ih in range(self.context_size + 1):
                if context_image[ih] is None: #to detect Nontype image and skip
                    flag_data_inner = 1
                    iv = random.randint(0, len(self.image_path)-1)
      
            pickle_values = self.aug_data_list[iv]
            #self._vis_dataset(image_fullsize, pickle_values)
            #if image_fullsize is None: #to detect Nontype image and skip
            #    flag_data_inner = 1
            #    iv = random.randint(0, len(self.image_path)-1)
                
            if len(pickle_values) != 0:
                list_rand = [random.randint(0, len(pickle_values)-1) for i in range(len(pickle_values))]            
                il = 0
                c_pose_check = 0
                while flag_data_inner == 0:
                    ir = list_rand[il]
                    
                    #ir = 0 #test only
                    #flag_back = 0 #test only
                    
                    #print(pickle_values[ir])
                    if flag_back == 0: #flag_back = 0 --> front-side, flag_back = 1 --> back-side
                        thres_data = pickle_values[ir]["bbox"][3] <= 224 and pickle_values[ir]["obj_detect"]
                        #print("front side!!")
                    else:
                        thres_data = pickle_values[ir]["bbox"][2] >= 224 and pickle_values[ir]["obj_detect"]
                        #print("back side!!")
                        
                    #if pickle_values[ir]["bbox"][1] <= 580 and pickle_values[ir]["bbox"][3] <= 580 and pickle_values[ir]["obj_detect"]:         
                    if thres_data:                   
                        #image_crop = image_fullsize[:, pickle_values[ir]["bbox"][0]:pickle_values[ir]["bbox"][1], pickle_values[ir]["bbox"][2]:pickle_values[ir]["bbox"][3]]
                        #image_crop = image_fullsize
                        
                        if 0 <= pickle_values[ir]["bbox"][0] and pickle_values[ir]["bbox"][0] < 224-1:
                            bbox_top = int(pickle_values[ir]["bbox"][0])
                        elif pickle_values[ir]["bbox"][0] < 0:
                            bbox_top = 0                        
                        else:
                            bbox_top = 223
                        if 0 <= pickle_values[ir]["bbox"][1] and pickle_values[ir]["bbox"][1] < 224-1:
                            bbox_bottom = int(pickle_values[ir]["bbox"][1])
                        elif pickle_values[ir]["bbox"][1] < 0:
                            bbox_bottom = 0                        
                        else:
                            bbox_bottom = 223
                        if 0 <= pickle_values[ir]["bbox"][2] and pickle_values[ir]["bbox"][2] < 224-1:
                            bbox_left = int(pickle_values[ir]["bbox"][2])
                        elif pickle_values[ir]["bbox"][2] < 0:
                            bbox_left = 0                        
                        else:
                            bbox_left = 223
                        if 0 <= pickle_values[ir]["bbox"][3] and pickle_values[ir]["bbox"][3] < 224-1:
                            bbox_right = int(pickle_values[ir]["bbox"][3])
                        elif pickle_values[ir]["bbox"][3] < 0:
                            bbox_right = 0                        
                        else:
                            bbox_right = 223
                                                                                
                        image_crop = image_fullsize[:, bbox_top:bbox_bottom, bbox_left:bbox_right]
                        #pose_obj = pickle_values[ir]["pose_median"]
                        
                        if flag_back == 0:
                            pose_obj = pickle_values[ir]["pose_median"]
                        else:
                            pose_obj = [-pickle_values[ir]["pose_median"][0], pickle_values[ir]["pose_median"][1], -pickle_values[ir]["pose_median"][2]]
                        
                        #prioritize "prompt_2", which is shorter than "prompt_1"
                        flag_text = 0
                        #c_length_check = 0
                        
                        if "prompt" in pickle_values[ir].keys():
                            ii = random.randint(0, len(pickle_values[ir]["prompt"])-1)
                            inst_obj = pickle_values[ir]["prompt"][ii]

                            if isinstance(inst_obj, list):
                                flag_text = 0
                            else:
                                flag_text = 1
                                                                                    
                        """
                        while flag_text_length == 0:
                            if "prompt" in pickle_values[ir].keys():
                                ii = random.randint(0, len(pickle_values[ir]["prompt"])-1)
                                #ii = 0 #test only
                                if c_length_check == 5:
                                    ii = len(pickle_values[ir]["prompt"])-1
                                inst_obj = pickle_values[ir]["prompt"][ii]
                                flag_text_length = 1
                            #else:
                            #    ii = random.randint(0, len(pickle_values[ir]["prompt_1"])-1)
                            #    #ii = 0 #test only
                            #    if c_length_check == 5:
                            #        ii = len(pickle_values[ir]["prompt_1"])-1                                
                            #    inst_obj = pickle_values[ir]["prompt_1"][ii]                            
                            
                            #inst_obj_list = inst_obj.lower().split(" ")
                            #if (len(inst_obj_list) == 1 or inst_obj_list[0] == "List") and c_length_check < 5:
                            #    flag_text_length = 0
                            #else:
                            #    flag_text_length = 1
                                
                            c_length_check += 1
                        """
                        #test = "a b '123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789999999999999999999999999999999"
                        #inst_obj_list = test.lower().split(" ")
                        #print(len(test), len(inst_obj.lower()))
                        
                        inst_obj_x = inst_obj
                        #print(inst_obj_x)
                        """
                        if len(inst_obj_list) <= 1 or len(inst_obj.lower()) > 100.0:
                            inst_obj_x = "xxxx"
                        else:
                            if len(inst_obj_list) < 20:
                                nmax = len(inst_obj_list)
                            else:
                                nmax = 20
                            
                            if inst_obj_list[1] == "go" and len(inst_obj_list) >= 5:
                                if  inst_obj_list[3] == "the":
                                    inst_obj_x = inst_obj_list[4]                            
                                    for text in inst_obj_list[5:nmax]:
                                        inst_obj_x = inst_obj_x + " " + text           
                                elif inst_obj_list[3] != "the":
                                    inst_obj_x = inst_obj_list[3]                            
                                    for text in inst_obj_list[4:nmax]:
                                        inst_obj_x = inst_obj_x + " " + text                                
                             
                            elif inst_obj_list[1] == "to" and len(inst_obj_list) >= 4:
                                if inst_obj_list[2] == "the":
                                    inst_obj_x = inst_obj_list[3]
                                    for text in inst_obj_list[4:nmax]:
                                        inst_obj_x = inst_obj_x + " " + text
                                elif inst_obj_list[2] != "the":
                                    inst_obj_x = inst_obj_list[2]
                                    for text in inst_obj_list[3:nmax]:
                                        inst_obj_x = inst_obj_x + " " + text                                                                                          
                            else:    
                                inst_obj_x = inst_obj_list[1]
                                for text in inst_obj_list[2:nmax]:
                                    inst_obj_x = inst_obj_x + " " + text
                                #print("else", inst_obj_x)  
                        """
                        #print("distance", pickle_values[ir]["pose_median"][0]**2 + pickle_values[ir]["pose_median"][2]**2, c_pose_check)
                        if pickle_values[ir]["pose_median"][0]**2 + pickle_values[ir]["pose_median"][2]**2 > 10.0**2 or flag_text == 0: 
                            c_pose_check += 1
                            if c_pose_check == 5:                  
                                flag_data_inner = 1
                                iv = random.randint(0, len(self.image_path)-1)
                            else:
                                flag_data_inner = 0                                
                        else:
                            flag_data_inner = 1 
                            flag_data = 1                                                                             
                    else:
                        il += 1
                        if il+1 > len(list_rand):
                            flag_data_inner = 1
                            iv = random.randint(0, len(self.image_path)-1)
                            #print("updating", iv)
            else:
                iv = random.randint(0, len(self.image_path)-1)
                #print("updating (no pickle)", iv)
            
        voffset = int(224.0*self.v_random*random.random())
        hoffset = int(224.0*self.h_random*random.random())
        
        image_obs_list = [] 
        if self.only_front:
            for ih in range(self.context_size + 1):
                image_obs_list.append(self._resize_norm(context_image[ih][:, voffset:224-voffset, hoffset:224-hoffset], self.image_size))                
        else:
            if flag_back == 0:
                for ih in range(self.context_size + 1):     
                    image_obs_list.append(self._resize_norm(context_image[ih][:, voffset:224-voffset, hoffset:224-hoffset], self.image_size))              
            else:
                for ih in range(self.context_size + 1):  
                    image_obs_list.append(self._resize_norm(context_image[ih][:, voffset:224-voffset, 224+hoffset:2*224-hoffset], self.image_size))                   
        image_obs = torch.cat(image_obs_list[::-1])      
        
        #image_obs = self._resize_norm(image_fullsize[:,:,0:580], self.image_size)
        image_crop = self._resize_norm(image_crop, self.image_size)        
        #print(image_obs.max(), image_obs.min(), image_obs.size(), image_crop.max(), image_crop.min(), image_crop.size(), torch.as_tensor(np.array(pose_obj), dtype=torch.float32))
               
        if random.random() > 0.5:
            #print(image_obs.shape, image_crop.shape)
            image_obs_r = torch.flip(image_obs, [2])
            image_crop_r = torch.flip(image_crop, [2])
            ob_pose_r = np.array((-pose_obj[0], pose_obj[2]))
        else:
            image_obs_r = image_obs
            image_crop_r = image_crop
            ob_pose_r = np.array((pose_obj[0], pose_obj[2]))        

        #test only
        #image_obs_r = image_obs
        #image_crop_r = image_crop
        #ob_pose_r = np.array((pose_obj[0], pose_obj[2]))  
        ob_pose_norm = ob_pose_r/self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing
        
        return (
            torch.as_tensor(image_obs_r, dtype=torch.float32),
            torch.as_tensor(image_crop_r, dtype=torch.float32),
            torch.as_tensor(ob_pose_r, dtype=torch.float32),
            inst_obj_x,
            torch.as_tensor(ob_pose_norm, dtype=torch.float32),
            #torch.as_tensor(np.array((pose_obj[0], pose_obj[2])), dtype=torch.float32),
        )         
