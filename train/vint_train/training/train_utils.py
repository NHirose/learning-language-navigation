import wandb
import os
import numpy as np
import yaml
from typing import List, Optional, Dict
from prettytable import PrettyTable
import tqdm
import itertools

from vint_train.visualizing.action_utils import visualize_traj_pred, plot_trajs_and_points
from vint_train.visualizing.distance_utils import visualize_dist_pred
from vint_train.visualizing.visualize_utils import to_numpy, from_numpy
from vint_train.training.logger import Logger
from vint_train.data.data_utils import VISUALIZATION_IMAGE_SIZE
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import clip

# LOAD DATA CONFIG
with open(os.path.join(os.path.dirname(__file__), "../data/data_config.yaml"), "r") as f:
    data_config = yaml.safe_load(f)
# POPULATE ACTION STATS
ACTION_STATS = {}
for key in data_config['action_stats']:
    ACTION_STATS[key] = np.array(data_config['action_stats'][key])

# Train utils for ViNT and GNM
def _compute_losses(
    dist_label: torch.Tensor,
    action_label: torch.Tensor,
    dist_pred: torch.Tensor,
    action_pred: torch.Tensor,
    alpha: float,
    learn_angle: bool,
    action_mask: torch.Tensor = None,
):
    """
    Compute losses for distance and action prediction.

    """
    dist_loss = F.mse_loss(dist_pred.squeeze(-1), dist_label.float())

    def action_reduce(unreduced_loss: torch.Tensor):
        # Reduce over non-batch dimensions to get loss per batch element
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
        return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

    # Mask out invalid inputs (for negatives, or when the distance between obs and goal is large)
    assert action_pred.shape == action_label.shape, f"{action_pred.shape} != {action_label.shape}"
    action_loss = action_reduce(F.mse_loss(action_pred, action_label, reduction="none"))

    action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
        action_pred[:, :, :2], action_label[:, :, :2], dim=-1
    ))
    multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(
        torch.flatten(action_pred[:, :, :2], start_dim=1),
        torch.flatten(action_label[:, :, :2], start_dim=1),
        dim=-1,
    ))

    results = {
        "dist_loss": dist_loss,
        "action_loss": action_loss,
        "action_waypts_cos_sim": action_waypts_cos_similairity,
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim,
    }

    if learn_angle:
        action_orien_cos_sim = action_reduce(F.cosine_similarity(
            action_pred[:, :, 2:], action_label[:, :, 2:], dim=-1
        ))
        multi_action_orien_cos_sim = action_reduce(F.cosine_similarity(
            torch.flatten(action_pred[:, :, 2:], start_dim=1),
            torch.flatten(action_label[:, :, 2:], start_dim=1),
            dim=-1,
            )
        )
        results["action_orien_cos_sim"] = action_orien_cos_sim
        results["multi_action_orien_cos_sim"] = multi_action_orien_cos_sim

    total_loss = alpha * 1e-2 * dist_loss + (1 - alpha) * action_loss
    results["total_loss"] = total_loss

    return results


def _log_data(
    i,
    epoch,
    num_batches,
    normalized,
    project_folder,
    num_images_log,
    loggers,
    obs_image,
    goal_image,
    action_pred,
    action_label,
    dist_pred,
    dist_label,
    goal_pos,
    dataset_index,
    use_wandb,
    mode,
    use_latest,
    wandb_log_freq=1,
    print_log_freq=1,
    image_log_freq=1,
    wandb_increment_step=True,
):
    """
    Log data to wandb and print to console.
    """
    data_log = {}
    for key, logger in loggers.items():
        if use_latest:
            data_log[logger.full_name()] = logger.latest()
            if i % print_log_freq == 0 and print_log_freq != 0:
                print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")
        else:
            data_log[logger.full_name()] = logger.average()
            if i % print_log_freq == 0 and print_log_freq != 0:
                print(f"(epoch {epoch}) {logger.full_name()} {logger.average()}")

    if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
        wandb.log(data_log, commit=wandb_increment_step)

    if image_log_freq != 0 and i % image_log_freq == 0:
        visualize_dist_pred(
            to_numpy(obs_image),
            to_numpy(goal_image),
            to_numpy(dist_pred),
            to_numpy(dist_label),
            mode,
            project_folder,
            epoch,
            num_images_log,
            use_wandb=use_wandb,
        )
        visualize_traj_pred(
            to_numpy(obs_image),
            to_numpy(goal_image),
            to_numpy(dataset_index),
            to_numpy(goal_pos),
            to_numpy(action_pred),
            to_numpy(action_label),
            mode,
            normalized,
            project_folder,
            epoch,
            num_images_log,
            use_wandb=use_wandb,
        )


def train(
    model: nn.Module,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    epoch: int,
    alpha: float = 0.5,
    learn_angle: bool = True,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
    use_tqdm: bool = True,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        learn_angle: whether to learn the angle of the action
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
        use_tqdm: whether to use tqdm
    """
    model.train()
    dist_loss_logger = Logger("dist_loss", "train", window_size=print_log_freq)
    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)
    action_waypts_cos_sim_logger = Logger(
        "action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    multi_action_waypts_cos_sim_logger = Logger(
        "multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    total_loss_logger = Logger("total_loss", "train", window_size=print_log_freq)
    loggers = {
        "dist_loss": dist_loss_logger,
        "action_loss": action_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim_logger,
        "total_loss": total_loss_logger,
    }

    if learn_angle:
        action_orien_cos_sim_logger = Logger(
            "action_orien_cos_sim", "train", window_size=print_log_freq
        )
        multi_action_orien_cos_sim_logger = Logger(
            "multi_action_orien_cos_sim", "train", window_size=print_log_freq
        )
        loggers["action_orien_cos_sim"] = action_orien_cos_sim_logger
        loggers["multi_action_orien_cos_sim"] = multi_action_orien_cos_sim_logger

    num_batches = len(dataloader)
    tqdm_iter = tqdm.tqdm(
        dataloader,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Training epoch {epoch}",
    )
    for i, data in enumerate(tqdm_iter):
        (
            obs_image,
            goal_image,
            action_label,
            dist_label,
            goal_pos,
            dataset_index,
            action_mask,
        ) = data

        obs_images = torch.split(obs_image, 3, dim=1)
        viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE)
        obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
        obs_image = torch.cat(obs_images, dim=1)

        viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE)
        
        goal_image = transform(goal_image).to(device)
        model_outputs = model(obs_image, goal_image)

        dist_label = dist_label.to(device)
        action_label = action_label.to(device)
        action_mask = action_mask.to(device)

        optimizer.zero_grad()
      
        dist_pred, action_pred = model_outputs

        losses = _compute_losses(
            dist_label=dist_label,
            action_label=action_label,
            dist_pred=dist_pred,
            action_pred=action_pred,
            alpha=alpha,
            learn_angle=learn_angle,
            action_mask=action_mask,
        )

        losses["total_loss"].backward()
        optimizer.step()

        for key, value in losses.items():
            if key in loggers:
                logger = loggers[key]
                logger.log_data(value.item())

        _log_data(
            i=i,
            epoch=epoch,
            num_batches=num_batches,
            normalized=normalized,
            project_folder=project_folder,
            num_images_log=num_images_log,
            loggers=loggers,
            obs_image=viz_obs_image,
            goal_image=viz_goal_image,
            action_pred=action_pred,
            action_label=action_label,
            dist_pred=dist_pred,
            dist_label=dist_label,
            goal_pos=goal_pos,
            dataset_index=dataset_index,
            wandb_log_freq=wandb_log_freq,
            print_log_freq=print_log_freq,
            image_log_freq=image_log_freq,
            use_wandb=use_wandb,
            mode="train",
            use_latest=True,
        )


def evaluate(
    eval_type: str,
    model: nn.Module,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    epoch: int = 0,
    alpha: float = 0.5,
    learn_angle: bool = True,
    num_images_log: int = 8,
    use_wandb: bool = True,
    eval_fraction: float = 1.0,
    use_tqdm: bool = True,

):
    """
    Evaluate the model on the given evaluation dataset.

    Args:
        eval_type (string): f"{data_type}_{eval_type}" (e.g. "recon_train", "gs_test", etc.)
        model (nn.Module): model to evaluate
        dataloader (DataLoader): dataloader for eval
        transform (transforms): transform to apply to images
        device (torch.device): device to use for evaluation
        project_folder (string): path to project folder
        epoch (int): current epoch
        alpha (float): weight for action loss
        learn_angle (bool): whether to learn the angle of the action
        num_images_log (int): number of images to log
        use_wandb (bool): whether to use wandb for logging
        eval_fraction (float): fraction of data to use for evaluation
        use_tqdm (bool): whether to use tqdm for logging
    """
    model.eval()
    dist_loss_logger = Logger("dist_loss", eval_type)
    action_loss_logger = Logger("action_loss", eval_type)
    action_waypts_cos_sim_logger = Logger("action_waypts_cos_sim", eval_type)
    multi_action_waypts_cos_sim_logger = Logger("multi_action_waypts_cos_sim", eval_type)
    total_loss_logger = Logger("total_loss", eval_type)
    loggers = {
        "dist_loss": dist_loss_logger,
        "action_loss": action_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim_logger,
        "total_loss": total_loss_logger,
    }

    if learn_angle:
        action_orien_cos_sim_logger = Logger("action_orien_cos_sim", eval_type)
        multi_action_orien_cos_sim_logger = Logger("multi_action_orien_cos_sim", eval_type)
        loggers["action_orien_cos_sim"] = action_orien_cos_sim_logger
        loggers["multi_action_orien_cos_sim"] = multi_action_orien_cos_sim_logger

    num_batches = len(dataloader)
    num_batches = max(int(num_batches * eval_fraction), 1)

    viz_obs_image = None
    with torch.no_grad():
        tqdm_iter = tqdm.tqdm(
            itertools.islice(dataloader, num_batches),
            total=num_batches,
            disable=not use_tqdm,
            dynamic_ncols=True,
            desc=f"Evaluating {eval_type} for epoch {epoch}",
        )
        for i, data in enumerate(tqdm_iter):
            (
                obs_image,
                goal_image,
                action_label,
                dist_label,
                goal_pos,
                dataset_index,
                action_mask,
            ) = data

            obs_images = torch.split(obs_image, 3, dim=1)
            viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE)
            obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
            obs_image = torch.cat(obs_images, dim=1)

            viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE)

            goal_image = transform(goal_image).to(device)
            model_outputs = model(obs_image, goal_image)

            dist_label = dist_label.to(device)
            action_label = action_label.to(device)
            action_mask = action_mask.to(device)

            dist_pred, action_pred = model_outputs

            losses = _compute_losses(
                dist_label=dist_label,
                action_label=action_label,
                dist_pred=dist_pred,
                action_pred=action_pred,
                alpha=alpha,
                learn_angle=learn_angle,
                action_mask=action_mask,
            )

            for key, value in losses.items():
                if key in loggers:
                    logger = loggers[key]
                    logger.log_data(value.item())

    # Log data to wandb/console, with visualizations selected from the last batch
    _log_data(
        i=i,
        epoch=epoch,
        num_batches=num_batches,
        normalized=normalized,
        project_folder=project_folder,
        num_images_log=num_images_log,
        loggers=loggers,
        obs_image=viz_obs_image,
        goal_image=viz_goal_image,
        action_pred=action_pred,
        action_label=action_label,
        goal_pos=goal_pos,
        dist_pred=dist_pred,
        dist_label=dist_label,
        dataset_index=dataset_index,
        use_wandb=use_wandb,
        mode=eval_type,
        use_latest=False,
        wandb_increment_step=False,
    )

    return dist_loss_logger.average(), action_loss_logger.average(), total_loss_logger.average()


# Train utils for NOMAD

def _compute_losses_nomad(
    ema_model,
    noise_scheduler,
    batch_obs_images,
    batch_goal_images,
    batch_dist_label: torch.Tensor,
    batch_action_label: torch.Tensor,
    device: torch.device,
    action_mask: torch.Tensor,
):
    """
    Compute losses for distance and action prediction.
    """

    pred_horizon = batch_action_label.shape[1]
    action_dim = batch_action_label.shape[2]

    model_output_dict = model_output(
        ema_model,
        noise_scheduler,
        batch_obs_images,
        batch_goal_images,
        pred_horizon,
        action_dim,
        num_samples=1,
        device=device,
    )
    uc_actions = model_output_dict['uc_actions']
    gc_actions = model_output_dict['gc_actions']
    gc_distance = model_output_dict['gc_distance']

    gc_dist_loss = F.mse_loss(gc_distance, batch_dist_label.unsqueeze(-1))

    def action_reduce(unreduced_loss: torch.Tensor):
        # Reduce over non-batch dimensions to get loss per batch element
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
        return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

    # Mask out invalid inputs (for negatives, or when the distance between obs and goal is large)
    assert uc_actions.shape == batch_action_label.shape, f"{uc_actions.shape} != {batch_action_label.shape}"
    assert gc_actions.shape == batch_action_label.shape, f"{gc_actions.shape} != {batch_action_label.shape}"

    uc_action_loss = action_reduce(F.mse_loss(uc_actions, batch_action_label, reduction="none"))
    gc_action_loss = action_reduce(F.mse_loss(gc_actions, batch_action_label, reduction="none"))

    uc_action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
        uc_actions[:, :, :2], batch_action_label[:, :, :2], dim=-1
    ))
    uc_multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(
        torch.flatten(uc_actions[:, :, :2], start_dim=1),
        torch.flatten(batch_action_label[:, :, :2], start_dim=1),
        dim=-1,
    ))

    gc_action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
        gc_actions[:, :, :2], batch_action_label[:, :, :2], dim=-1
    ))
    gc_multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(
        torch.flatten(gc_actions[:, :, :2], start_dim=1),
        torch.flatten(batch_action_label[:, :, :2], start_dim=1),
        dim=-1,
    ))

    results = {
        "uc_action_loss": uc_action_loss,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_similairity,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim,
        "gc_dist_loss": gc_dist_loss,
        "gc_action_loss": gc_action_loss,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_similairity,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim,
    }

    return results

def _compute_losses_lnp(
    ema_model,
    batch_obs_images,
    batch_goal_images,
    batch_dist_label: torch.Tensor,
    device: torch.device,
):
    """
    Compute losses for distance and action prediction.
    """

    pred_horizon = batch_dist_label.shape[1]
    action_dim = batch_dist_label.shape[2]

    model_output_dict = model_output(
        ema_model,
        noise_scheduler,
        batch_obs_images,
        batch_goal_images,
        pred_horizon,
        action_dim,
        num_samples=1,
        device=device,
    )
    uc_actions = model_output_dict['uc_actions']
    gc_actions = model_output_dict['gc_actions']
    gc_distance = model_output_dict['gc_distance']

    gc_dist_loss = F.mse_loss(gc_distance, batch_dist_label.unsqueeze(-1))

    def action_reduce(unreduced_loss: torch.Tensor):
        # Reduce over non-batch dimensions to get loss per batch element
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
        return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

    # Mask out invalid inputs (for negatives, or when the distance between obs and goal is large)
    assert uc_actions.shape == batch_dist_label.shape, f"{uc_actions.shape} != {batch_dist_label.shape}"
    assert gc_actions.shape == batch_dist_label.shape, f"{gc_actions.shape} != {batch_dist_label.shape}"

    uc_action_loss = action_reduce(F.mse_loss(uc_actions, batch_dist_label, reduction="none"))
    gc_action_loss = action_reduce(F.mse_loss(gc_actions, batch_dist_label, reduction="none"))

    uc_action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
        uc_actions[:, :, :2], batch_dist_label[:, :, :2], dim=-1
    ))
    uc_multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(
        torch.flatten(uc_actions[:, :, :2], start_dim=1),
        torch.flatten(batch_dist_label[:, :, :2], start_dim=1),
        dim=-1,
    ))

    gc_action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
        gc_actions[:, :, :2], batch_dist_label[:, :, :2], dim=-1
    ))
    gc_multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(
        torch.flatten(gc_actions[:, :, :2], start_dim=1),
        torch.flatten(batch_dist_label[:, :, :2], start_dim=1),
        dim=-1,
    ))

    results = {
        "uc_action_loss": uc_action_loss,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_similairity,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim,
        "gc_dist_loss": gc_dist_loss,
        "gc_action_loss": gc_action_loss,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_similairity,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim,
    }

    return results
    
def sinc_apx(angle):
    return torch.sin(3.141592*angle + 0.000000001)/(3.141592*angle + 0.000000001)
        
def twist_to_pose_diff_torch(v, w, dt):
    """integrate 2D twist to get pose difference.

    Assuming constant velocity during time period `dt`.

    Args:
        v (float): velocity
        w (float): angular velocity
        dt (float): time delta

    """

    theta = -w  * dt
    z = v * dt * sinc_apx(-theta / np.pi)
    x = -v * dt * sinc_apx(-theta / (2 * np.pi)) * torch.sin(-theta / 2)
    return x, z, theta

def robot_pos_model(linear_vel, angular_vel):
    bs, chorizon = linear_vel.shape
    device = linear_vel.device
    px_ref = torch.zeros(bs).to(device)
    pz_ref = torch.zeros(bs).to(device)
    ry_ref = torch.zeros(bs).to(device)

    #print(chorizon)
    for j in range(chorizon):
        ry_ref = ry_ref - angular_vel[:, j] * 0.33
        pz_ref = pz_ref + linear_vel[:, j] * 0.33 * sinc_apx(-ry_ref/3.1415)
        px_ref = px_ref - linear_vel[:, j] * 0.33 * sinc_apx(-ry_ref/(2*3.1415)) * torch.sin(-ry_ref / 2.0)
    #print("x, z, yaw", px_ref, pz_ref, ry_ref)    
    return px_ref, pz_ref, ry_ref

def robot_pos_model_fix(linear_vel, angular_vel):
    bs, chorizon = linear_vel.shape
    device = linear_vel.device

    px = []
    pz = []
    pyaw = []
    Tacc = torch.eye(4, 4).unsqueeze(0).repeat(bs,1,1).to(device)
    for i in range(chorizon):
        x, z, yaw = twist_to_pose_diff_torch(linear_vel[:, i], angular_vel[:, i], 0.333)
        Todom = torch.zeros((bs, 4, 4)).to(device)
        Todom[:, 0, 0] = torch.cos(yaw)
        Todom[:, 0, 2] = torch.sin(yaw)
        Todom[:, 1, 1] = 1.0
        Todom[:, 2, 0] = -torch.sin(yaw)
        Todom[:, 2, 2] = torch.cos(yaw)
        Todom[:, 0, 3] = x #weighting for position
        Todom[:, 2, 3] = z #weighting for position
        Todom[:, 3, 3] = 1.0        
        
        #Tacc = Tacc @ Todom
        Tacc = torch.matmul(Tacc, Todom)
        
        #Taccd = Tacc.clone()
        #Taccd[:, 0, 3] = 2.0*Tacc.clone()[:, 0, 3]
        #Taccd[:, 2, 3] = 2.0*Tacc.clone()[:, 2, 3]        
        pyaw.append(torch.arctan(Tacc[:, 0, 2]/(Tacc[:, 0, 0] + 0.000000001)))        
        px.append(Tacc[:, 0, 3])
        pz.append(Tacc[:, 2, 3])        
    return px, pz, pyaw
                        
def robot_pos_model_vis(linear_vel, angular_vel):
    bs, chorizon = linear_vel.shape
    device = linear_vel.device
    px_ref = torch.zeros(bs).to(device)
    pz_ref = torch.zeros(bs).to(device)
    ry_ref = torch.zeros(bs).to(device)
    
    px = []
    pz = []
    px.append(px_ref.unsqueeze(1))
    pz.append(pz_ref.unsqueeze(1))
        
    for j in range(chorizon):
        ry_ref = ry_ref - angular_vel[:, j] * 0.33
        pz_ref = pz_ref + linear_vel[:, j] * 0.33 * sinc_apx(-ry_ref/3.1415)
        px_ref = px_ref - linear_vel[:, j] * 0.33 * sinc_apx(-ry_ref/(2*3.1415)) * torch.sin(-ry_ref / 2.0)
    
        px.append(px_ref.unsqueeze(1))
        pz.append(pz_ref.unsqueeze(1))
        
    return torch.cat(px, axis=1), torch.cat(pz, axis=1)
    

def train_lelan(
    model: nn.Module,
    ema_model: EMAModel,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    #noise_scheduler: DDPMScheduler,
    #goal_mask_prob: float,
    project_folder: str,
    epoch: int,
    alpha: float = 1e-4,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    #goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    model.train()
    model.eval_text_encoder()
    #print("eval model for text encoder!!")
    num_batches = len(dataloader)
    """
    uc_action_loss_logger = Logger("uc_action_loss", "train", window_size=print_log_freq)
    uc_action_waypts_cos_sim_logger = Logger(
        "uc_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    uc_multi_action_waypts_cos_sim_logger = Logger(
        "uc_multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    gc_dist_loss_logger = Logger("gc_dist_loss", "train", window_size=print_log_freq)
    gc_action_loss_logger = Logger("gc_action_loss", "train", window_size=print_log_freq)
    gc_action_waypts_cos_sim_logger = Logger(
        "gc_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    gc_multi_action_waypts_cos_sim_logger = Logger(
        "gc_multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    """
    total_loss_logger = Logger("total loss", "train", window_size=print_log_freq)    
    pose_loss_logger = Logger("pose loss", "train", window_size=print_log_freq)
    smooth_loss_logger = Logger(
        "smooth loss", "train", window_size=print_log_freq
    )    
    loggers = {
        "total loss": total_loss_logger,    
        "pose loss": pose_loss_logger,
        "vel smooth loss": smooth_loss_logger,
    }
    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_images, 
                goal_image,
                obj_poses,
                obj_inst,
                goal_pos_norm,                
            ) = data
            
            #print(obs_image.size())
            obs_images_list = torch.split(obs_images, 3, dim=1)
            obs_image = obs_images_list[-1]              
            
            batch_viz_obs_images = TF.resize((127.5*obs_image + 127.5).type(torch.uint8), VISUALIZATION_IMAGE_SIZE[::-1])
            batch_viz_goal_images = TF.resize((127.5*goal_image + 127.5).type(torch.uint8), VISUALIZATION_IMAGE_SIZE[::-1])
            #batch_viz_obs_images = TF.resize(0.5*(obs_image + 1.0), VISUALIZATION_IMAGE_SIZE[::-1])
            #batch_viz_goal_images = TF.resize(0.5*(goal_image + 1.0), VISUALIZATION_IMAGE_SIZE[::-1])
                        
            #mid = (127.5*obs_image + 127.5)            
            #print("vis", batch_viz_obs_images.max(), batch_viz_obs_images.min(), batch_viz_obs_images.median(), batch_viz_obs_images.type(), batch_viz_obs_images.size())
            #print("train", obs_image.max(), obs_image.min(), obs_image.median(), obs_image.type(), obs_image.size())           
            #print("mid", mid.max(), mid.min(), mid.median(), mid.type(), mid.size())                    
            batch_obs_images = obs_image.to(device)
            #batch_goal_images = goal_image.to(device)
            batch_obj_poses = obj_poses.to(device)
            
            #print(obj_inst)
            batch_obj_inst = clip.tokenize(obj_inst, truncate=True).to(device)          
            
            with torch.no_grad():  
                feat_text = model("text_encoder", inst_ref=batch_obj_inst)
            
            #TODO contrastive loss
            #batch_goal_images = goal_image.to(device)
            #feat_image = model("vision_encoder_clip", goal_img=batch_goal_images)
            
            #obs_images = torch.split(obs_image, 3, dim=1)
            #batch_viz_obs_images = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            #batch_viz_goal_images = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            #batch_obs_images = [transform(obs) for obs in obs_images]
            #batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)
            #batch_goal_images = transform(goal_image).to(device)
            #action_mask = action_mask.to(device)

            #B = actions.shape[0]
            B = batch_obs_images.shape[0]
            
            # Generate random goal mask
            #goal_mask = (torch.rand((B,)) < goal_mask_prob).long().to(device)
            #obsgoal_cond = model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=goal_mask)
            
            #print("batch_obs_images", batch_obs_images.size())
            #print(feat_text.dtype)
            obsgoal_cond = model("vision_encoder", obs_img=batch_obs_images, feat_text = feat_text.to(dtype=torch.float32))
            #obsgoal_cond = model("vision_encoder", obs_img=batch_obs_images, inst_ref = batch_obj_inst)
            #print("feat", obsgoal_cond.size())
            
            """            
            # Get distance label
            distance = distance.float().to(device)

            print("actions", actions.shape)
            deltas = get_delta(actions)
            print("deltas", deltas.shape)            
            ndeltas = normalize_data(deltas, ACTION_STATS)
            print("ndeltas", ndeltas.shape)             
            naction = from_numpy(ndeltas).to(device)
            print("naction", naction.shape)                     
            assert naction.shape[-1] == 2, "action dim must be 2"
            """
            # Predict distance
            #dist_pred = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
            linear_vel, angular_vel = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
            #print("vel", linear_vel.size())
            #print(linear_vel.max(), linear_vel.min(), angular_vel.max(), angular_vel.min())
            #print(linear_vel.size(), angular_vel.size(), linear_vel.device, angular_vel.device)
            
            #bs, vs = angular_vel.size()
            #linear_vel = 0.5*torch.ones(bs, vs).float().cuda()
            #angular_vel = 0.1*torch.ones(bs, vs).float().cuda()            
            #print("vel size", bs, vs)
            px_ref_list, pz_ref_list, ry_ref_list = robot_pos_model_fix(linear_vel, angular_vel)
            px_ref = px_ref_list[-1]
            pz_ref = pz_ref_list[-1]
            ry_ref = ry_ref_list[-1]
            #print("velocity", linear_vel.mean(), angular_vel.mean())
            #print("last pose", px_ref.mean(), pz_ref.mean())       
            #print("goal_pos", batch_obj_poses[:,0].mean(), batch_obj_poses[:,1].mean(), batch_obj_poses.size())                 
            #batch_px_list, batch_pz_list = robot_pos_model_vis(linear_vel, angular_vel)
            #x_seq = to_numpy(batch_px_list[i])
            #z_seq = to_numpy(batch_pz_list[i])
       
            #print("new", px_ref[0].size(), pz_ref[0].size(), ry_ref[0].size(), len(px_ref), len(pz_ref), len(ry_ref))
            #print("new", px_ref_list[-1][0], pz_ref_list[-1][0], ry_ref_list[-1][0])            
            #px_ref, pz_ref, ry_ref = robot_pos_model(linear_vel, angular_vel)
            #print("old", px_ref[0], pz_ref[0], ry_ref[0])            
            last_poses = torch.cat((px_ref.unsqueeze(1), pz_ref.unsqueeze(1)), axis=1)
            #print("x, z, yaw", px_ref.size(),  pz_ref.size(),  ry_ref.size(), obj_poses.size(), last_poses.device)

            """
            for i in range(24):
                plt.plot(px_ref_list[i][0].cpu().numpy(), pz_ref_list[i][0].cpu().numpy(), color='blue', marker = "x")
        
            plt.plot(x_seq, z_seq, marker = 'o', color='blue')
            
            plt.show()  
            """
                                
            dist_loss = nn.functional.mse_loss(last_poses, batch_obj_poses)   
            diff_loss = nn.functional.mse_loss(linear_vel[:,:-1], linear_vel[:,1:]) + nn.functional.mse_loss(angular_vel[:,:-1], angular_vel[:,1:]) 
            
            #print(torch.sum((last_poses - batch_obj_poses)**2, axis=1))            
            #print("dist_loss", dist_loss)        
            #dist_loss = nn.functional.mse_loss(dist_pred.squeeze(-1), distance)
            #dist_loss = (dist_loss * (1 - goal_mask.float())).mean() / (1e-2 +(1 - goal_mask.float()).mean())

            """
            # Sample noise to add to actions
            noise = torch.randn(naction.shape, device=device)

            # Sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (B,), device=device
            ).long()
            print(noise_scheduler.config.num_train_timesteps, B, timesteps.size())

            # Add noise to the clean images according to the noise magnitude at each diffusion iteration
            noisy_action = noise_scheduler.add_noise(
                naction, noise, timesteps)
            print("noisy_action", noisy_action.size())
            print("timesteps", timesteps.size())
            print("obsgoal_cond", obsgoal_cond.size())            
                        
            # Predict the noise residual
            noise_pred = model("noise_pred_net", sample=noisy_action, timestep=timesteps, global_cond=obsgoal_cond)

            print("noise_pred", noise_pred.size(), "noise", noise.size())

            def action_reduce(unreduced_loss: torch.Tensor):
                # Reduce over non-batch dimensions to get loss per batch element
                while unreduced_loss.dim() > 1:
                    unreduced_loss = unreduced_loss.mean(dim=-1)
                assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
                return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

            # L2 loss
            diffusion_loss = action_reduce(F.mse_loss(noise_pred, noise, reduction="none"))
            """
            # Total loss
            #loss = alpha * dist_loss + (1-alpha) * diffusion_loss # mse between ground truth noise and predicted noise
            loss = dist_loss + 1.0*diff_loss
            #print("dist_loss", dist_loss, "diff_loss", diff_loss)

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update Exponential Moving Average of the model weights
            ema_model.step(model)

            # Logging
            
            loss_cpu = loss.item()
            tepoch.set_postfix(loss=loss_cpu)
            wandb.log({"total loss": loss_cpu})
            wandb.log({"pose loss": dist_loss.item()})
            wandb.log({"vel smooth loss": diff_loss.item()})

            if i % print_log_freq == 0:
                """
                losses = _compute_losses_nomad(
                            ema_model.averaged_model,
                            noise_scheduler,
                            batch_obs_images,
                            batch_goal_images,
                            distance.to(device),
                            actions.to(device),
                            device,
                            action_mask.to(device),
                        )
                """
                losses = {}
                losses['total loss'] = loss_cpu
                losses['pose loss'] = dist_loss.item()
                losses['vel smooth loss'] = diff_loss.item()                 
                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value)
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)

            if image_log_freq != 0 and i % image_log_freq == 0:
                #print("test", linear_vel.max(), linear_vel.min(), angular_vel.max(), angular_vel.min())
                visualize_lelan_estimation(
                    batch_viz_obs_images,
                    batch_viz_goal_images,
                    obj_poses,
                    obj_inst,
                    linear_vel.cpu(),
                    angular_vel.cpu(),
                    last_poses.cpu(),
                    "train",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,                    
                    use_wandb,
                )

def train_lelan_col(
    model: nn.Module,
    ema_model: EMAModel,
    ema_model_nomad: EMAModel,    
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    noise_scheduler: DDPMScheduler,
    #goal_mask_prob: float,
    project_folder: str,
    weight_col_loss: float,    
    epoch: int,
    alpha: float = 1e-4,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    #goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    model.train()
    model.eval_text_encoder()
    ema_model_nomad = ema_model_nomad.averaged_model
    ema_model_nomad.eval()    
    num_batches = len(dataloader)
    """
    uc_action_loss_logger = Logger("uc_action_loss", "train", window_size=print_log_freq)
    uc_action_waypts_cos_sim_logger = Logger(
        "uc_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    uc_multi_action_waypts_cos_sim_logger = Logger(
        "uc_multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    gc_dist_loss_logger = Logger("gc_dist_loss", "train", window_size=print_log_freq)
    gc_action_loss_logger = Logger("gc_action_loss", "train", window_size=print_log_freq)
    gc_action_waypts_cos_sim_logger = Logger(
        "gc_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    gc_multi_action_waypts_cos_sim_logger = Logger(
        "gc_multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    """
    total_loss_logger = Logger("total loss", "train", window_size=print_log_freq)    
    pose_loss_logger = Logger("pose loss", "train", window_size=print_log_freq)
    smooth_loss_logger = Logger("smooth loss", "train", window_size=print_log_freq)    
    col_loss_logger = Logger("col loss", "train", window_size=print_log_freq)       
    loggers = {
        "total loss": total_loss_logger,    
        "pose loss": pose_loss_logger,
        "vel smooth loss": smooth_loss_logger,
        "col loss": col_loss_logger,        
    }
    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_images, 
                goal_image,
                goal_pos,
                obj_inst,
                goal_pos_norm,                
            ) = data
            
            obs_images_list = torch.split(obs_images, 3, dim=1)
            obs_image = obs_images_list[-1]              
            
            batch_viz_obs_images = TF.resize((127.5*obs_image + 127.5).type(torch.uint8), VISUALIZATION_IMAGE_SIZE[::-1])
            batch_viz_goal_images = TF.resize((127.5*goal_image + 127.5).type(torch.uint8), VISUALIZATION_IMAGE_SIZE[::-1])
                                          
            batch_obs_current = obs_image.to(device)

            batch_goal_pos = goal_pos.to(device)
            batch_goal_pos_norm = goal_pos_norm.to(device)      
                        
            batch_obs_images = [transform(TF.resize(obs, (96, 96), antialias=True)) for obs in obs_images_list]
            batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)
            batch_goal_images = transform(TF.resize(goal_image, (96, 96), antialias=True)).to(device)
            
            batch_obj_inst = clip.tokenize(obj_inst, truncate=True).to(device)          
            
            with torch.no_grad():  
                feat_text = model("text_encoder", inst_ref=batch_obj_inst)
            
            B = batch_obs_images.shape[0]
            action_mask = torch.ones(B).to(device)
                        
            # split into batches
            batch_obs_images_list = torch.split(batch_obs_images, B, dim=0)
            batch_goal_images_list = torch.split(batch_goal_images, B, dim=0)

            with torch.no_grad():
                select_traj = supervision_from_diffusion_action_distribution(
                    ema_model_nomad,
                    noise_scheduler,
                    batch_obs_images,
                    batch_goal_images,
                    batch_viz_obs_images,
                    batch_viz_goal_images,
                    #actions,
                    #distance,
                    batch_goal_pos_norm,
                    device,
                    #eval_type,
                    project_folder,
                    epoch,
                    B,
                    i,                
                    30,
                    use_wandb,
                    )    

            with torch.no_grad():
                feat_text = model("text_encoder", inst_ref=batch_obj_inst)
                                                
            obsgoal_cond = model("vision_encoder", obs_img=batch_obs_images, feat_text = feat_text.to(dtype=torch.float32), current_img=batch_obs_current)
            linear_vel, angular_vel = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
            
            px_ref_list, pz_ref_list, ry_ref_list = robot_pos_model_fix(linear_vel, angular_vel)
            px_ref = px_ref_list[-1]
            pz_ref = pz_ref_list[-1]
            ry_ref = ry_ref_list[-1]
            last_poses = torch.cat((px_ref.unsqueeze(1), pz_ref.unsqueeze(1)), axis=1)

            #transformation from camera coordinate to robot coordinate
            px_ref_listx = []
            pz_ref_listx = []
            for it in range(8):
                px_ref_listx.append(px_ref_list[it].unsqueeze(1).unsqueeze(2))
                pz_ref_listx.append(pz_ref_list[it].unsqueeze(1).unsqueeze(2))
            traj_policy = torch.concat((torch.concat(pz_ref_listx, axis=1), -torch.concat(px_ref_listx, axis=1)), axis=2)
                                
            dist_loss = nn.functional.mse_loss(last_poses, batch_goal_pos)   
            diff_loss = nn.functional.mse_loss(linear_vel[:,:-1], linear_vel[:,1:]) + nn.functional.mse_loss(angular_vel[:,:-1], angular_vel[:,1:]) 
            col_loss = nn.functional.mse_loss(traj_policy, 0.12*select_traj) #0.12 is de-normalization
            
            loss = 1.0*dist_loss + 1.0*diff_loss + weight_col_loss*col_loss

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update Exponential Moving Average of the model weights
            ema_model.step(model)

            # Logging
            
            loss_cpu = loss.item()
            tepoch.set_postfix(loss=loss_cpu)
            wandb.log({"total loss": loss_cpu})
            wandb.log({"pose loss": dist_loss.item()})
            wandb.log({"vel smooth loss": diff_loss.item()})
            wandb.log({"col loss": col_loss.item()})
            
            if i % print_log_freq == 0:
                losses = {}
                losses['total loss'] = loss_cpu
                losses['pose loss'] = dist_loss.item()
                losses['vel smooth loss'] = diff_loss.item()                 
                losses['col loss'] = col_loss.item()       
                                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value)
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)

            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_lelan_estimation(
                    batch_viz_obs_images,
                    batch_viz_goal_images,
                    goal_pos,
                    obj_inst,
                    linear_vel.cpu(),
                    angular_vel.cpu(),
                    last_poses.cpu(),
                    "train",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,                    
                    use_wandb,
                )

def train_nomad(
    model: nn.Module,
    ema_model: EMAModel,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    noise_scheduler: DDPMScheduler,
    goal_mask_prob: float,
    project_folder: str,
    epoch: int,
    alpha: float = 1e-4,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    model.train()
    num_batches = len(dataloader)

    uc_action_loss_logger = Logger("uc_action_loss", "train", window_size=print_log_freq)
    uc_action_waypts_cos_sim_logger = Logger(
        "uc_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    uc_multi_action_waypts_cos_sim_logger = Logger(
        "uc_multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    gc_dist_loss_logger = Logger("gc_dist_loss", "train", window_size=print_log_freq)
    gc_action_loss_logger = Logger("gc_action_loss", "train", window_size=print_log_freq)
    gc_action_waypts_cos_sim_logger = Logger(
        "gc_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    gc_multi_action_waypts_cos_sim_logger = Logger(
        "gc_multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    loggers = {
        "uc_action_loss": uc_action_loss_logger,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_sim_logger,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim_logger,
        "gc_dist_loss": gc_dist_loss_logger,
        "gc_action_loss": gc_action_loss_logger,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_sim_logger,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim_logger,
    }
    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_image, 
                goal_image,
                actions,
                distance,
                goal_pos,
                dataset_idx,
                action_mask, 
            ) = data

            
            obs_images = torch.split(obs_image, 3, dim=1)
            batch_viz_obs_images = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            batch_viz_goal_images = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            batch_obs_images = [transform(obs) for obs in obs_images]
            batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)
            batch_goal_images = transform(goal_image).to(device)
            action_mask = action_mask.to(device)

            B = actions.shape[0]

            # Generate random goal mask
            goal_mask = (torch.rand((B,)) < goal_mask_prob).long().to(device)
            obsgoal_cond = model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=goal_mask)
            
            # Get distance label
            distance = distance.float().to(device)

            print("actions", actions.shape)
            deltas = get_delta(actions)
            print("deltas", deltas.shape)            
            ndeltas = normalize_data(deltas, ACTION_STATS)
            print("ndeltas", ndeltas.shape)             
            naction = from_numpy(ndeltas).to(device)
            print("naction", naction.shape)                     
            assert naction.shape[-1] == 2, "action dim must be 2"

            # Predict distance
            dist_pred = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
            dist_loss = nn.functional.mse_loss(dist_pred.squeeze(-1), distance)
            dist_loss = (dist_loss * (1 - goal_mask.float())).mean() / (1e-2 +(1 - goal_mask.float()).mean())

            # Sample noise to add to actions
            noise = torch.randn(naction.shape, device=device)

            # Sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (B,), device=device
            ).long()
            print(noise_scheduler.config.num_train_timesteps, B, timesteps.size())

            # Add noise to the clean images according to the noise magnitude at each diffusion iteration
            noisy_action = noise_scheduler.add_noise(
                naction, noise, timesteps)
            print("noisy_action", noisy_action.size())
            print("timesteps", timesteps.size())
            print("obsgoal_cond", obsgoal_cond.size())            
                        
            # Predict the noise residual
            noise_pred = model("noise_pred_net", sample=noisy_action, timestep=timesteps, global_cond=obsgoal_cond)

            print("noise_pred", noise_pred.size(), "noise", noise.size())

            def action_reduce(unreduced_loss: torch.Tensor):
                # Reduce over non-batch dimensions to get loss per batch element
                while unreduced_loss.dim() > 1:
                    unreduced_loss = unreduced_loss.mean(dim=-1)
                assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
                return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

            # L2 loss
            diffusion_loss = action_reduce(F.mse_loss(noise_pred, noise, reduction="none"))
            
            # Total loss
            loss = alpha * dist_loss + (1-alpha) * diffusion_loss # mse between ground truth noise and predicted noise

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update Exponential Moving Average of the model weights
            ema_model.step(model)

            # Logging
            
            loss_cpu = loss.item()
            tepoch.set_postfix(loss=loss_cpu)
            wandb.log({"total_loss": loss_cpu})
            wandb.log({"dist_loss": dist_loss.item()})
            wandb.log({"diffusion_loss": diffusion_loss.item()})


            if i % print_log_freq == 0:
                losses = _compute_losses_nomad(
                            ema_model.averaged_model,
                            noise_scheduler,
                            batch_obs_images,
                            batch_goal_images,
                            distance.to(device),
                            actions.to(device),
                            device,
                            action_mask.to(device),
                        )
                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value.item())
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)

            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_diffusion_action_distribution(
                    ema_model.averaged_model,
                    noise_scheduler,
                    batch_obs_images,
                    batch_goal_images,
                    batch_viz_obs_images,
                    batch_viz_goal_images,
                    actions,
                    distance,
                    goal_pos,
                    device,
                    "train",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,
                    use_wandb,
                )


def evaluate_lelan(
    eval_type: str,
    ema_model: EMAModel,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    #noise_scheduler: DDPMScheduler,
    #goal_mask_prob: float,
    project_folder: str,
    epoch: int,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    eval_fraction: float = 0.25,
    use_wandb: bool = True,
):
    
    """
    Evaluate the model on the given evaluation dataset.

    Args:
        eval_type (string): f"{data_type}_{eval_type}" (e.g. "recon_train", "gs_test", etc.)
        ema_model (nn.Module): exponential moving average version of model to evaluate
        dataloader (DataLoader): dataloader for eval
        transform (transforms): transform to apply to images
        device (torch.device): device to use for evaluation
        noise_scheduler: noise scheduler to evaluate with 
        project_folder (string): path to project folder
        epoch (int): current epoch    total_loss_logger = Logger("total loss", "train", window_size=print_log_freq)    
    pose_loss_logger = Logger("pose loss", "train", window_size=print_log_freq)
    smooth_loss_logger = Logger(
        "smooth loss", "train", window_size=print_log_freq
    )    
    loggers = {
        "total loss": total_loss_logger,    
        "pose loss": pose_loss_logger,
        "vel smooth loss": smooth_loss_logger,
    }
        print_log_freq (int): how often to print logs 
        wandb_log_freq (int): how often to log to wandb
        image_log_freq (int): how often to log images
        alpha (float): weight for action loss
        num_images_log (int): number of images to log
        eval_fraction (float): fraction of data to use for evaluation
        use_wandb (bool): whether to use wandb for logging
    """
    #goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    #ema_model = ema_model.averaged_model #TODO for diffusion policy
    ema_model.eval()
    
    num_batches = len(dataloader)
    """
    uc_action_loss_logger = Logger("uc_action_loss", eval_type, window_size=print_log_freq)
    uc_action_waypts_cos_sim_logger = Logger(
        "uc_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    uc_multi_action_waypts_cos_sim_logger = Logger(
        "uc_multi_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    gc_dist_loss_logger = Logger("gc_dist_loss", eval_type, window_size=print_log_freq)
    gc_action_loss_logger = Logger("gc_action_loss", eval_type, window_size=print_log_freq)
    gc_action_waypts_cos_sim_logger = Logger(
        "gc_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    gc_multi_action_waypts_cos_sim_logger = Logger(
        "gc_multi_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    loggers = {
        "uc_action_loss": uc_action_loss_logger,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_sim_logger,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim_logger,
        "gc_dist_loss": gc_dist_loss_logger,
        "gc_action_loss": gc_action_loss_logger,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_sim_logger,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim_logger,
    }
    """
    total_loss_logger = Logger("total loss", eval_type, window_size=print_log_freq)    
    pose_loss_logger = Logger("pose loss", eval_type, window_size=print_log_freq)
    smooth_loss_logger = Logger("smooth loss", eval_type, window_size=print_log_freq)     
    loggers = {
        "total loss": total_loss_logger,    
        "pose loss": pose_loss_logger,
        "vel smooth loss": smooth_loss_logger,
    }    
    num_batches = max(int(num_batches * eval_fraction), 1)

    all_total = 0.0
    all_dist = 0.0
    all_diff = 0.0
    
    count_batch = 0
    data_size = 0
    with tqdm.tqdm(
        itertools.islice(dataloader, num_batches), 
        total=num_batches, 
        dynamic_ncols=True, 
        desc=f"Evaluating {eval_type} for epoch {epoch}", 
        leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_images, 
                goal_image,
                obj_poses,
                obj_inst,
                goal_pos_norm,
            ) = data
            
            obs_images_list = torch.split(obs_images, 3, dim=1)
            obs_image = obs_images_list[-1]       
            
            batch_viz_obs_images = TF.resize((127.5*obs_image + 127.5).type(torch.uint8), VISUALIZATION_IMAGE_SIZE[::-1])
            batch_viz_goal_images = TF.resize((127.5*goal_image + 127.5).type(torch.uint8), VISUALIZATION_IMAGE_SIZE[::-1])
            #batch_viz_obs_images = TF.resize(0.5*(obs_image + 1.0), VISUALIZATION_IMAGE_SIZE[::-1])
            #batch_viz_goal_images = TF.resize(0.5*(goal_image + 1.0), VISUALIZATION_IMAGE_SIZE[::-1])
                        
            #mid = (127.5*obs_image + 127.5)            
            #print("vis", batch_viz_obs_images.max(), batch_viz_obs_images.min(), batch_viz_obs_images.median(), batch_viz_obs_images.type(), batch_viz_obs_images.size())
            #print("train", obs_image.max(), obs_image.min(), obs_image.median(), obs_image.type(), obs_image.size())           
            #print("mid", mid.max(), mid.min(), mid.median(), mid.type(), mid.size())                    
            batch_obs_images = obs_image.to(device)
            #batch_goal_images = goal_image.to(device)
            batch_obj_poses = obj_poses.to(device)
            
            #print(obj_inst)
            
            with torch.no_grad():
                batch_obj_inst = clip.tokenize(obj_inst, truncate=True).to(device)       
                #print(batch_obj_inst.size())     
                feat_text = ema_model("text_encoder", inst_ref=batch_obj_inst)
                #print(feat_text.size(), feat_text.dtype, batch_obj_inst.size(), batch_obj_inst.dtype)
                
                #feat_text_16 = feat_text.to(torch.float16)    
                #feat_text = feat_text_16.to(torch.float32)   
                #print(feat_text_16.dtype)                
                                
                B = batch_obs_images.shape[0]

                #print(batch_obs_images.dtype, batch_obs_images.size(), batch_obs_images.min(), batch_obs_images.max())                    
                #print("batch_obs_images", batch_obs_images.size(), num_batches, eval_fraction, B)
                obsgoal_cond = ema_model("vision_encoder", obs_img=batch_obs_images, feat_text = feat_text.to(dtype=torch.float32))

                #obsgoal_cond_16 = obsgoal_cond.to(torch.float16)    
                #obsgoal_cond_16 = obsgoal_cond_16.to(torch.float32)   
            
                #print("before flatten", obsgoal_cond.size())
                #obsgoal_cond = obsgoal_cond.flatten(start_dim=1)
                #print("after flatten", obsgoal_cond.size())
                        
                linear_vel, angular_vel = ema_model("dist_pred_net", obsgoal_cond=obsgoal_cond)
                #print(linear_vel.size(), angular_vel.size())
                            
                #print(obj_inst[0])
                #print(feat_text[0])
                #print(batch_obj_inst.dtype, batch_obs_images.dtype, feat_text.dtype, obsgoal_cond.dtype)
                px_ref_list, pz_ref_list, ry_ref_list = robot_pos_model_fix(linear_vel, angular_vel)
                px_ref = px_ref_list[-1]
                pz_ref = pz_ref_list[-1]
                ry_ref = ry_ref_list[-1]
                #px_ref, pz_ref, ry_ref = robot_pos_model(linear_vel, angular_vel)
                                                    
            last_poses = torch.cat((px_ref.unsqueeze(1), pz_ref.unsqueeze(1)), axis=1)
                        
            #print(last_poses.size(), batch_obj_poses.size())
            """
            last_poses_select = []
            batch_obj_poses_select = []
            #print("before", last_poses.size(), batch_obj_poses.size())
                        
            for ir in range(len(obj_inst)):
                text_split = obj_inst[ir].split(" ")
                #print(text_split)
                flag_remove = 0
                #print("door" in text_split)
                if ("door" in text_split) or ("chair" in text_split) or ("wall" in text_split) or ("glass" in text_split) or ("device" in text_split) or ("window" in text_split) or ("handle" in text_split) or ("cabinet" in text_split):
                    flag_remove = 1
                else:
                    last_poses_select.append(last_poses[ir].unsqueeze(0))
                    batch_obj_poses_select.append(batch_obj_poses[ir].unsqueeze(0))
            
            last_poses = torch.cat(last_poses_select, axis=0)
            batch_obj_poses = torch.cat(batch_obj_poses_select, axis=0)                                
            #print("after", last_poses.size(), batch_obj_poses.size())            
            """
            dist_loss = nn.functional.mse_loss(last_poses, batch_obj_poses)   
            diff_loss = nn.functional.mse_loss(linear_vel[:,:-1], linear_vel[:,1:]) + nn.functional.mse_loss(angular_vel[:,:-1], angular_vel[:,1:]) 
                        
            # Logging
            loss_cpu = dist_loss.item()
            tepoch.set_postfix(loss=loss_cpu)

            #wandb.log({"diffusion_eval_loss (random masking)": dist_loss})
            #wandb.log({"diffusion_eval_loss (no masking)": dist_loss})
            #wandb.log({"diffusion_eval_loss (goal masking)": dist_loss})
            wandb.log({"total_eval_loss": (dist_loss + 1.0*diff_loss).item()})
            wandb.log({"dist_eval_loss": dist_loss.item()})
            wandb.log({"diff_eval_loss": diff_loss.item()})

            all_total += (dist_loss + 1.0*diff_loss).item()
            all_dist += dist_loss.item()
            all_diff += diff_loss.item()
            count_batch += 1.0
            data_size += B
            if i % print_log_freq == 0 and print_log_freq != 0:
                """
                losses = _compute_losses_nomad(
                            ema_model,
                            noise_scheduler,
                            batch_obs_images,
                            batch_goal_images,
                            distance.to(device),
                            actions.to(device),
                            device,
                            action_mask.to(device),
                        )
                """
                #losses = {}
                #losses['uc_actions'] = 0.0
                #losses['gc_actions'] = 0.0
                #losses['gc_distance'] = 0.0   
                losses = {}
                losses['total loss'] = loss_cpu
                losses['pose loss'] = dist_loss.item()
                losses['vel smooth loss'] = diff_loss.item()             
                                                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value)
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)
            
            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_lelan_estimation(
                    batch_viz_obs_images,
                    batch_viz_goal_images,
                    obj_poses,
                    obj_inst,
                    linear_vel.cpu(),
                    angular_vel.cpu(),
                    last_poses.cpu(),
                    eval_type,
                    project_folder,
                    epoch,
                    num_images_log,
                    30,                    
                    use_wandb,
                )                
    print(eval_type, "total loss:", all_total/count_batch, "dist loss:", all_dist/count_batch, "diff loss:", all_diff/count_batch, "batch count:", count_batch, "data size:", data_size)

def evaluate_lelan_col(
    eval_type: str,
    ema_model: EMAModel,
    ema_model_nomad: EMAModel,    
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    noise_scheduler: DDPMScheduler,
    #goal_mask_prob: float,
    project_folder: str,
    weight_col_loss: float,    
    epoch: int,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    eval_fraction: float = 0.25,
    use_wandb: bool = True,
):
    
    """
    Evaluate the model on the given evaluation dataset.

    Args:
        eval_type (string): f"{data_type}_{eval_type}" (e.g. "recon_train", "gs_test", etc.)
        ema_model (nn.Module): exponential moving average version of model to evaluate
        dataloader (DataLoader): dataloader for eval
        transform (transforms): transform to apply to images
        device (torch.device): device to use for evaluation
        noise_scheduler: noise scheduler to evaluate with 
        project_folder (string): path to project folder
        epoch (int): current epoch    total_loss_logger = Logger("total loss", "train", window_size=print_log_freq)    
    pose_loss_logger = Logger("pose loss", "train", window_size=print_log_freq)
    smooth_loss_logger = Logger(
        "smooth loss", "train", window_size=print_log_freq
    )    
    loggers = {
        "total loss": total_loss_logger,    
        "pose loss": pose_loss_logger,
        "vel smooth loss": smooth_loss_logger,
    }
        print_log_freq (int): how often to print logs 
        wandb_log_freq (int): how often to log to wandb
        image_log_freq (int): how often to log images
        alpha (float): weight for action loss
        num_images_log (int): number of images to log
        eval_fraction (float): fraction of data to use for evaluation
        use_wandb (bool): whether to use wandb for logging
    """

    ema_model.eval()
    ema_model_nomad = ema_model_nomad.averaged_model
    ema_model_nomad.eval()       
    num_batches = len(dataloader)
    """
    uc_action_loss_logger = Logger("uc_action_loss", eval_type, window_size=print_log_freq)
    uc_action_waypts_cos_sim_logger = Logger(
        "uc_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    uc_multi_action_waypts_cos_sim_logger = Logger(
        "uc_multi_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    gc_dist_loss_logger = Logger("gc_dist_loss", eval_type, window_size=print_log_freq)
    gc_action_loss_logger = Logger("gc_action_loss", eval_type, window_size=print_log_freq)
    gc_action_waypts_cos_sim_logger = Logger(
        "gc_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    gc_multi_action_waypts_cos_sim_logger = Logger(
        "gc_multi_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    loggers = {
        "uc_action_loss": uc_action_loss_logger,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_sim_logger,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim_logger,
        "gc_dist_loss": gc_dist_loss_logger,
        "gc_action_loss": gc_action_loss_logger,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_sim_logger,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim_logger,
    }
    """
    total_loss_logger = Logger("total loss", eval_type, window_size=print_log_freq)    
    pose_loss_logger = Logger("pose loss", eval_type, window_size=print_log_freq)
    smooth_loss_logger = Logger("smooth loss", eval_type, window_size=print_log_freq)    
    col_loss_logger = Logger("col loss", eval_type, window_size=print_log_freq) 
    loggers = {
        "total loss": total_loss_logger,    
        "pose loss": pose_loss_logger,
        "vel smooth loss": smooth_loss_logger,
        "col loss": col_loss_logger,        
    }    
    num_batches = max(int(num_batches * eval_fraction), 1)

    all_total = 0.0
    all_dist = 0.0
    all_diff = 0.0
    all_col = 0.0
        
    count_batch = 0
    data_size = 0
    with tqdm.tqdm(
        itertools.islice(dataloader, num_batches), 
        total=num_batches, 
        dynamic_ncols=True, 
        desc=f"Evaluating {eval_type} for epoch {epoch}", 
        leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_images, 
                goal_image,
                goal_pos,
                obj_inst,
                goal_pos_norm,
            ) = data
            
            obs_images_list = torch.split(obs_images, 3, dim=1)
            obs_image = obs_images_list[-1]              
            
            batch_viz_obs_images = TF.resize((127.5*obs_image + 127.5).type(torch.uint8), VISUALIZATION_IMAGE_SIZE[::-1])
            batch_viz_goal_images = TF.resize((127.5*goal_image + 127.5).type(torch.uint8), VISUALIZATION_IMAGE_SIZE[::-1])
                                          
            batch_obs_current = obs_image.to(device)

            goal_pos = goal_pos.to(device)
            goal_pos_norm = goal_pos_norm.to(device)      
                        
            batch_obs_images = [transform(TF.resize(obs, (96, 96), antialias=True)) for obs in obs_images_list]
            batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)
            batch_goal_images = transform(TF.resize(goal_image, (96, 96), antialias=True)).to(device)
            
            B = batch_obs_images.shape[0]
            action_mask = torch.ones(B).to(device)
                        
            # split into batches
            batch_obs_images_list = torch.split(batch_obs_images, B, dim=0)
            batch_goal_images_list = torch.split(batch_goal_images, B, dim=0)

            with torch.no_grad():
                select_traj = supervision_from_diffusion_action_distribution(
                    ema_model_nomad,
                    noise_scheduler,
                    batch_obs_images,
                    batch_goal_images,
                    batch_viz_obs_images,
                    batch_viz_goal_images,
                    #actions,
                    #distance,
                    goal_pos_norm,
                    device,
                    #eval_type,
                    project_folder,
                    epoch,
                    B,
                    i,                
                    30,
                    use_wandb,
                    )                
            
            with torch.no_grad():
                batch_obj_inst = clip.tokenize(obj_inst, truncate=True).to(device)         
                feat_text = ema_model("text_encoder", inst_ref=batch_obj_inst)       
                                
                B = batch_obs_images.shape[0]

                obsgoal_cond = ema_model("vision_encoder", obs_img=batch_obs_images, feat_text = feat_text.to(dtype=torch.float32), current_img=batch_obs_current)
                linear_vel, angular_vel = ema_model("dist_pred_net", obsgoal_cond=obsgoal_cond)

                px_ref_list, pz_ref_list, ry_ref_list = robot_pos_model_fix(linear_vel, angular_vel)
                px_ref = px_ref_list[-1]
                pz_ref = pz_ref_list[-1]
                ry_ref = ry_ref_list[-1]
                                                    
            last_poses = torch.cat((px_ref.unsqueeze(1), pz_ref.unsqueeze(1)), axis=1)
            px_ref_listx = []
            pz_ref_listx = []
            for it in range(8):            
                px_ref_listx.append(px_ref_list[it].unsqueeze(1).unsqueeze(2))
                pz_ref_listx.append(pz_ref_list[it].unsqueeze(1).unsqueeze(2))
            traj_policy = torch.concat((torch.concat(px_ref_listx, axis=1), torch.concat(pz_ref_listx, axis=1)), axis=2)
                                    
            dist_loss = nn.functional.mse_loss(last_poses, goal_pos)   
            diff_loss = nn.functional.mse_loss(linear_vel[:,:-1], linear_vel[:,1:]) + nn.functional.mse_loss(angular_vel[:,:-1], angular_vel[:,1:]) 
            col_loss = nn.functional.mse_loss(traj_policy, 0.12*select_traj) #0.12 is de-normalization
            
            loss = 1.0*dist_loss + 1.0*diff_loss + weight_col_loss*col_loss
                                    
            # Logging
            loss_cpu = dist_loss.item()
            tepoch.set_postfix(loss=loss_cpu)

            #wandb.log({"diffusion_eval_loss (random masking)": dist_loss})
            #wandb.log({"diffusion_eval_loss (no masking)": dist_loss})
            #wandb.log({"diffusion_eval_loss (goal masking)": dist_loss})
            wandb.log({"total_eval_loss": (dist_loss + 1.0*diff_loss + weight_col_loss*col_loss).item()})
            wandb.log({"dist_eval_loss": dist_loss.item()})
            wandb.log({"diff_eval_loss": diff_loss.item()})
            wandb.log({"col_eval_loss": col_loss.item()})
            
            all_total += (dist_loss + 1.0*diff_loss + weight_col_loss*col_loss).item()
            all_dist += dist_loss.item()
            all_diff += diff_loss.item()
            all_col += col_loss.item()            
            count_batch += 1.0
            data_size += B
            if i % print_log_freq == 0 and print_log_freq != 0:
                """
                losses = _compute_losses_nomad(
                            ema_model,
                            noise_scheduler,
                            batch_obs_images,
                            batch_goal_images,
                            distance.to(device),
                            actions.to(device),
                            device,
                            action_mask.to(device),
                        )
                """
                #losses = {}
                #losses['uc_actions'] = 0.0
                #losses['gc_actions'] = 0.0
                #losses['gc_distance'] = 0.0   
                losses = {}
                losses['total loss'] = loss_cpu
                losses['pose loss'] = dist_loss.item()
                losses['vel smooth loss'] = diff_loss.item()             
                losses['col loss'] = col_loss.item()       
                                                                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value)
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)
            
            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_lelan_estimation(
                    batch_viz_obs_images,
                    batch_viz_goal_images,
                    goal_pos,
                    obj_inst,
                    linear_vel.cpu(),
                    angular_vel.cpu(),
                    last_poses.cpu(),
                    eval_type,
                    project_folder,
                    epoch,
                    num_images_log,
                    30,                    
                    use_wandb,
                )                
    print(eval_type, "total loss:", all_total/count_batch, "dist loss:", all_dist/count_batch, "diff loss:", all_diff/count_batch, "col loss:", all_col/count_batch, "batch count:", count_batch, "data size:", data_size)


def evaluate_nomad(
    eval_type: str,
    ema_model: EMAModel,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    noise_scheduler: DDPMScheduler,
    goal_mask_prob: float,
    project_folder: str,
    epoch: int,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    eval_fraction: float = 0.25,
    use_wandb: bool = True,
):
    """
    Evaluate the model on the given evaluation dataset.

    Args:
        eval_type (string): f"{data_type}_{eval_type}" (e.g. "recon_train", "gs_test", etc.)
        ema_model (nn.Module): exponential moving average version of model to evaluate
        dataloader (DataLoader): dataloader for eval
        transform (transforms): transform to apply to images
        device (torch.device): device to use for evaluation
        noise_scheduler: noise scheduler to evaluate with 
        project_folder (string): path to project folder
        epoch (int): current epoch
        print_log_freq (int): how often to print logs 
        wandb_log_freq (int): how often to log to wandb
        image_log_freq (int): how often to log images
        alpha (float): weight for action loss
        num_images_log (int): number of images to log
        eval_fraction (float): fraction of data to use for evaluation
        use_wandb (bool): whether to use wandb for logging
    """
    goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    ema_model = ema_model.averaged_model
    ema_model.eval()
    
    num_batches = len(dataloader)

    uc_action_loss_logger = Logger("uc_action_loss", eval_type, window_size=print_log_freq)
    uc_action_waypts_cos_sim_logger = Logger(
        "uc_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    uc_multi_action_waypts_cos_sim_logger = Logger(
        "uc_multi_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    gc_dist_loss_logger = Logger("gc_dist_loss", eval_type, window_size=print_log_freq)
    gc_action_loss_logger = Logger("gc_action_loss", eval_type, window_size=print_log_freq)
    gc_action_waypts_cos_sim_logger = Logger(
        "gc_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    gc_multi_action_waypts_cos_sim_logger = Logger(
        "gc_multi_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    loggers = {
        "uc_action_loss": uc_action_loss_logger,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_sim_logger,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim_logger,
        "gc_dist_loss": gc_dist_loss_logger,
        "gc_action_loss": gc_action_loss_logger,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_sim_logger,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim_logger,
    }
    num_batches = max(int(num_batches * eval_fraction), 1)

    with tqdm.tqdm(
        itertools.islice(dataloader, num_batches), 
        total=num_batches, 
        dynamic_ncols=True, 
        desc=f"Evaluating {eval_type} for epoch {epoch}", 
        leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_image, 
                goal_image,
                actions,
                distance,
                goal_pos,
                dataset_idx,
                action_mask,
            ) = data
            
            obs_images = torch.split(obs_image, 3, dim=1)
            batch_viz_obs_images = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            batch_viz_goal_images = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            batch_obs_images = [transform(obs) for obs in obs_images]
            batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)
            batch_goal_images = transform(goal_image).to(device)
            action_mask = action_mask.to(device)

            B = actions.shape[0]

            # Generate random goal mask
            rand_goal_mask = (torch.rand((B,)) < goal_mask_prob).long().to(device)
            goal_mask = torch.ones_like(rand_goal_mask).long().to(device)
            no_mask = torch.zeros_like(rand_goal_mask).long().to(device)

            rand_mask_cond = ema_model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=rand_goal_mask)

            obsgoal_cond = ema_model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=no_mask)
            obsgoal_cond = obsgoal_cond.flatten(start_dim=1)

            goal_mask_cond = ema_model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=goal_mask)

            distance = distance.to(device)

            deltas = get_delta(actions)
            ndeltas = normalize_data(deltas, ACTION_STATS)
            naction = from_numpy(ndeltas).to(device)
            assert naction.shape[-1] == 2, "action dim must be 2"

            # Sample noise to add to actions
            noise = torch.randn(naction.shape, device=device)

            # Sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (B,), device=device
            ).long()

            noisy_actions = noise_scheduler.add_noise(
                naction, noise, timesteps)

            ### RANDOM MASK ERROR ###
            # Predict the noise residual
            rand_mask_noise_pred = ema_model("noise_pred_net", sample=noisy_actions, timestep=timesteps, global_cond=rand_mask_cond)
            
            # L2 loss
            rand_mask_loss = nn.functional.mse_loss(rand_mask_noise_pred, noise)
            
            ### NO MASK ERROR ###
            # Predict the noise residual
            no_mask_noise_pred = ema_model("noise_pred_net", sample=noisy_actions, timestep=timesteps, global_cond=obsgoal_cond)
            
            # L2 loss
            no_mask_loss = nn.functional.mse_loss(no_mask_noise_pred, noise)

            ### GOAL MASK ERROR ###
            # predict the noise residual
            goal_mask_noise_pred = ema_model("noise_pred_net", sample=noisy_actions, timestep=timesteps, global_cond=goal_mask_cond)
            
            # L2 loss
            goal_mask_loss = nn.functional.mse_loss(goal_mask_noise_pred, noise)
            
            # Logging
            loss_cpu = rand_mask_loss.item()
            tepoch.set_postfix(loss=loss_cpu)

            wandb.log({"diffusion_eval_loss (random masking)": rand_mask_loss})
            wandb.log({"diffusion_eval_loss (no masking)": no_mask_loss})
            wandb.log({"diffusion_eval_loss (goal masking)": goal_mask_loss})

            if i % print_log_freq == 0 and print_log_freq != 0:
                losses = _compute_losses_nomad(
                            ema_model,
                            noise_scheduler,
                            batch_obs_images,
                            batch_goal_images,
                            distance.to(device),
                            actions.to(device),
                            device,
                            action_mask.to(device),
                        )
                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value.item())
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)

            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_diffusion_action_distribution(
                    ema_model,
                    noise_scheduler,
                    batch_obs_images,
                    batch_goal_images,
                    batch_viz_obs_images,
                    batch_viz_goal_images,
                    actions,
                    distance,
                    goal_pos,
                    device,
                    eval_type,
                    project_folder,
                    epoch,
                    num_images_log,
                    30,
                    use_wandb,
                )


# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

def get_delta(actions):
    # append zeros to first action
    ex_actions = np.concatenate([np.zeros((actions.shape[0],1,actions.shape[-1])), actions], axis=1)
    delta = ex_actions[:,1:] - ex_actions[:,:-1]
    return delta

def get_action(diffusion_output, action_stats=ACTION_STATS):
    # diffusion_output: (B, 2*T+1, 1)
    # return: (B, T-1)
    device = diffusion_output.device
    ndeltas = diffusion_output
    ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)
    ndeltas = to_numpy(ndeltas)
    ndeltas = unnormalize_data(ndeltas, action_stats)
    actions = np.cumsum(ndeltas, axis=1)
    return from_numpy(actions).to(device)


def model_output(
    model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    pred_horizon: int,
    action_dim: int,
    num_samples: int,
    device: torch.device,
):
    goal_mask = torch.ones((batch_goal_images.shape[0],)).long().to(device)
    obs_cond = model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=goal_mask)
    # obs_cond = obs_cond.flatten(start_dim=1)
    obs_cond = obs_cond.repeat_interleave(num_samples, dim=0)

    no_mask = torch.zeros((batch_goal_images.shape[0],)).long().to(device)
    obsgoal_cond = model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=no_mask)
    # obsgoal_cond = obsgoal_cond.flatten(start_dim=1)  
    obsgoal_cond = obsgoal_cond.repeat_interleave(num_samples, dim=0)

    # initialize action from Gaussian noise
    noisy_diffusion_output = torch.randn(
        (len(obs_cond), pred_horizon, action_dim), device=device)
    diffusion_output = noisy_diffusion_output


    for k in noise_scheduler.timesteps[:]:
        # predict noise
        noise_pred = model(
            "noise_pred_net",
            sample=diffusion_output,
            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
            global_cond=obs_cond
        )

        # inverse diffusion step (remove noise)
        diffusion_output = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=diffusion_output
        ).prev_sample

    uc_actions = get_action(diffusion_output, ACTION_STATS)

    # initialize action from Gaussian noise
    noisy_diffusion_output = torch.randn(
        (len(obs_cond), pred_horizon, action_dim), device=device)
    diffusion_output = noisy_diffusion_output

    for k in noise_scheduler.timesteps[:]:
        # predict noise
        noise_pred = model(
            "noise_pred_net",
            sample=diffusion_output,
            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
            global_cond=obsgoal_cond
        )

        # inverse diffusion step (remove noise)
        diffusion_output = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=diffusion_output
        ).prev_sample
    obsgoal_cond = obsgoal_cond.flatten(start_dim=1)
    gc_actions = get_action(diffusion_output, ACTION_STATS)
    gc_distance = model("dist_pred_net", obsgoal_cond=obsgoal_cond)

    return {
        'uc_actions': uc_actions,
        'gc_actions': gc_actions,
        'gc_distance': gc_distance,
    }

def supervision_from_diffusion_action_distribution(
    ema_model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    batch_viz_obs_images: torch.Tensor,
    batch_viz_goal_images: torch.Tensor,
    batch_goal_pos: torch.Tensor,
    device: torch.device,
    project_folder: str,
    epoch: int,
    num_images_log: int,
    it_num: int,    
    num_samples: int = 30,
    use_wandb: bool = True,
):
    """Plot samples from the exploration model."""

    max_batch_size = batch_obs_images.shape[0]

    num_images_log = min(num_images_log, batch_obs_images.shape[0], batch_goal_images.shape[0], batch_goal_pos.shape[0])
    batch_obs_images = batch_obs_images[:num_images_log]
    batch_goal_images = batch_goal_images[:num_images_log]
    
    #wandb_list = []
    pred_horizon = 8
    action_dim = 2
    
    # split into batches
    batch_obs_images_list = torch.split(batch_obs_images, max_batch_size, dim=0)
    batch_goal_images_list = torch.split(batch_goal_images, max_batch_size, dim=0)

    #uc_actions_list = []
    gc_actions_torch_list = []    
    gc_actions_list = []
    #gc_distances_list = []

    for obs, goal in zip(batch_obs_images_list, batch_goal_images_list):
        model_output_dict = model_output(
            ema_model,
            noise_scheduler,
            obs,
            goal,
            pred_horizon,
            action_dim,
            num_samples,
            device,
        )
        gc_actions_torch_list.append(model_output_dict['gc_actions'])        
        #gc_actions_list.append(to_numpy(model_output_dict['gc_actions']))

    gc_actions_torch_list = torch.concat(gc_actions_torch_list, axis=0)    
    #gc_actions_list = np.concatenate(gc_actions_list, axis=0)

    gc_actions_torch_list = torch.split(gc_actions_torch_list, num_samples, dim=0)    
    #gc_actions_list = np.split(gc_actions_list, num_images_log, axis=0)
    
    select_traj_list = []
    for i in range(num_images_log):
        #gc_actions = gc_actions_list[i]
        gc_actions_torch = gc_actions_torch_list[i]
        #print("gc_actions_torch", gc_actions_torch.size())
        gc_actions_torch_cat = torch.concat(torch.split(gc_actions_torch, 1, dim=1), axis=0).squeeze(1)  
        #print("gc_actions_torch_cat", gc_actions_torch_cat.size())
        
        batch_goal_pos_i = torch.tensor([batch_goal_pos[i][1], -batch_goal_pos[i][0]])
        #print(batch_goal_pos_i.size())        
        device = gc_actions_torch_cat.get_device()
        
        batch_goal_pos_repeat = batch_goal_pos_i.unsqueeze(0).repeat(num_samples*8, 1).to(device)
        #print(batch_goal_pos_repeat)
        #batch_goal_pos_repeat = batch_goal_pos[i].unsqueeze(0).repeat(num_samples*8, 1)    
        #print(batch_goal_pos_repeat)                   
        #print("after", batch_goal_pos_repeat_i)
        #print("before", batch_goal_pos_repeat) 
        #print(batch_goal_pos_repeat.size(), gc_actions_torch_cat.size())
        traj_id_all = torch.argmin(torch.sum((batch_goal_pos_repeat - gc_actions_torch_cat)**2, axis=1))
        traj_id = traj_id_all % num_samples
        #print(traj_id, traj_id_all)

        select_traj_list.append(gc_actions_torch[traj_id:traj_id+1])
        """
        fig, ax = plt.subplots(1, 3)
        traj_list = np.concatenate([
            gc_actions_torch[traj_id:traj_id+1].cpu().numpy(),        
            gc_actions_torch.cpu().numpy(),
            #action_label[None],
        ], axis=0)

        traj_colors = ["red"] * 1 + ["green"] * len(gc_actions_torch)
        traj_alphas = [1.0] * 1 + [0.1] * len(gc_actions_torch)
                
        # make points numpy array of robot positions (0, 0) and goal positions
        point_list = [np.array([0, 0]), to_numpy(batch_goal_pos_i)]
        point_colors = ["green", "orange"]
        point_alphas = [1.0, 1.0]

        plot_trajs_and_points_noriaki(
            ax[0],
            traj_list,
            point_list,
            traj_colors,
            point_colors,
            traj_labels=None,
            point_labels=None,
            quiver_freq=0,
            traj_alphas=traj_alphas,
            point_alphas=point_alphas, 
        )
        
        obs_image = to_numpy(batch_viz_obs_images[i])
        goal_image = to_numpy(batch_viz_goal_images[i])
        # move channel to last dimension
        obs_image = np.moveaxis(obs_image, 0, -1)
        goal_image = np.moveaxis(goal_image, 0, -1)
        #print("obs_image", obs_image.max(), obs_image.min())
        #print("goal_image", goal_image.max(), goal_image.min())        
        ax[1].imshow(obs_image)
        ax[2].imshow(goal_image)

        # set title
        ax[0].set_title(f"diffusion action predictions")
        ax[1].set_title(f"observation")
        
        # make the plot large
        fig.set_size_inches(18.5, 10.5)

        plt.savefig("sample_" + str(it_num) + "_" + str(i) + ".png")        
        plt.close(fig)
        """
    return torch.concat(select_traj_list, axis=0)

def visualize_diffusion_action_distribution(
    ema_model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    batch_viz_obs_images: torch.Tensor,
    batch_viz_goal_images: torch.Tensor,
    batch_action_label: torch.Tensor,
    batch_distance_labels: torch.Tensor,
    batch_goal_pos: torch.Tensor,
    device: torch.device,
    eval_type: str,
    project_folder: str,
    epoch: int,
    num_images_log: int,
    num_samples: int = 30,
    use_wandb: bool = True,
):
    """Plot samples from the exploration model."""

    visualize_path = os.path.join(
        project_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "action_sampling_prediction",
    )
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)

    max_batch_size = batch_obs_images.shape[0]

    num_images_log = min(num_images_log, batch_obs_images.shape[0], batch_goal_images.shape[0], batch_action_label.shape[0], batch_goal_pos.shape[0])
    batch_obs_images = batch_obs_images[:num_images_log]
    batch_goal_images = batch_goal_images[:num_images_log]
    batch_action_label = batch_action_label[:num_images_log]
    batch_goal_pos = batch_goal_pos[:num_images_log]
    
    wandb_list = []

    pred_horizon = batch_action_label.shape[1]
    action_dim = batch_action_label.shape[2]

    # split into batches
    batch_obs_images_list = torch.split(batch_obs_images, max_batch_size, dim=0)
    batch_goal_images_list = torch.split(batch_goal_images, max_batch_size, dim=0)

    uc_actions_list = []
    gc_actions_list = []
    gc_distances_list = []

    for obs, goal in zip(batch_obs_images_list, batch_goal_images_list):
        model_output_dict = model_output(
            ema_model,
            noise_scheduler,
            obs,
            goal,
            pred_horizon,
            action_dim,
            num_samples,
            device,
        )
        uc_actions_list.append(to_numpy(model_output_dict['uc_actions']))
        gc_actions_list.append(to_numpy(model_output_dict['gc_actions']))
        gc_distances_list.append(to_numpy(model_output_dict['gc_distance']))

    # concatenate
    uc_actions_list = np.concatenate(uc_actions_list, axis=0)
    gc_actions_list = np.concatenate(gc_actions_list, axis=0)
    gc_distances_list = np.concatenate(gc_distances_list, axis=0)

    # split into actions per observation
    uc_actions_list = np.split(uc_actions_list, num_images_log, axis=0)
    gc_actions_list = np.split(gc_actions_list, num_images_log, axis=0)
    gc_distances_list = np.split(gc_distances_list, num_images_log, axis=0)

    gc_distances_avg = [np.mean(dist) for dist in gc_distances_list]
    gc_distances_std = [np.std(dist) for dist in gc_distances_list]

    assert len(uc_actions_list) == len(gc_actions_list) == num_images_log

    np_distance_labels = to_numpy(batch_distance_labels)

    for i in range(num_images_log):
        fig, ax = plt.subplots(1, 3)
        uc_actions = uc_actions_list[i]
        gc_actions = gc_actions_list[i]
        action_label = to_numpy(batch_action_label[i])

        traj_list = np.concatenate([
            uc_actions,
            gc_actions,
            action_label[None],
        ], axis=0)
        # traj_labels = ["r", "GC", "GC_mean", "GT"]
        traj_colors = ["red"] * len(uc_actions) + ["green"] * len(gc_actions) + ["magenta"]
        traj_alphas = [0.1] * (len(uc_actions) + len(gc_actions)) + [1.0]

        # make points numpy array of robot positions (0, 0) and goal positions
        point_list = [np.array([0, 0]), to_numpy(batch_goal_pos[i])]
        point_colors = ["green", "red"]
        point_alphas = [1.0, 1.0]

        plot_trajs_and_points(
            ax[0],
            traj_list,
            point_list,
            traj_colors,
            point_colors,
            traj_labels=None,
            point_labels=None,
            quiver_freq=0,
            traj_alphas=traj_alphas,
            point_alphas=point_alphas, 
        )
        
        obs_image = to_numpy(batch_viz_obs_images[i])
        goal_image = to_numpy(batch_viz_goal_images[i])
        # move channel to last dimension
        obs_image = np.moveaxis(obs_image, 0, -1)
        goal_image = np.moveaxis(goal_image, 0, -1)
        ax[1].imshow(obs_image)
        ax[2].imshow(goal_image)

        # set title
        ax[0].set_title(f"diffusion action predictions")
        ax[1].set_title(f"observation")
        ax[2].set_title(f"goal: label={np_distance_labels[i]} gc_dist={gc_distances_avg[i]:.2f}±{gc_distances_std[i]:.2f}")
        
        # make the plot large
        fig.set_size_inches(18.5, 10.5)

        save_path = os.path.join(visualize_path, f"sample_{i}.png")
        plt.savefig(save_path)
        wandb_list.append(wandb.Image(save_path))
        plt.close(fig)
    if len(wandb_list) > 0 and use_wandb:
        wandb.log({f"{eval_type}_action_samples": wandb_list}, commit=False)

def visualize_lelan_estimation(
    batch_viz_obs_images: torch.Tensor,
    batch_viz_goal_images: torch.Tensor,
    obj_poses: torch.Tensor,
    obj_inst: torch.Tensor,
    linear_vel: torch.Tensor,
    angular_vel: torch.Tensor,
    last_poses: torch.Tensor,
    eval_type: str,    
    project_folder: str,
    epoch: int,
    num_images_log: int,
    num_samples: int = 30,    
    use_wandb: bool = True,
):
    """Plot samples from the exploration model."""

    visualize_path = os.path.join(
        project_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "action_sampling_prediction",
    )
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)

    #max_batch_size = batch_obs_images.shape[0]

    num_images_log = min(num_images_log, batch_viz_obs_images.shape[0], batch_viz_goal_images.shape[0], obj_poses.shape[0], last_poses.shape[0])
    #batch_obs_images = batch_viz_obs_images[:num_images_log]
    #batch_goal_images = batch_viz_goal_images[:num_images_log]
    #batch_action_label = obj_poses[:num_images_log]
    #batch_goal_pos = last_poses[:num_images_log]
    
    batch_linear_vel = linear_vel[:num_images_log]
    batch_angular_vel = angular_vel[:num_images_log]
    
    #print("testtest", linear_vel.max(), linear_vel.min(), angular_vel.max(), angular_vel.min())
    
    #batch_px_list, batch_pz_list = robot_pos_model_vis(batch_linear_vel, batch_angular_vel)
    px_list, pz_list, ry_list = robot_pos_model_fix(batch_linear_vel, batch_angular_vel)
    
    px_list_a = []
    pz_list_a = []
    for px_v in px_list:
        px_list_a.append(px_v.unsqueeze(1))
    for pz_v in pz_list:
        pz_list_a.append(pz_v.unsqueeze(1))
            
    batch_px_list = torch.cat(px_list_a, axis=1)
    batch_pz_list = torch.cat(pz_list_a, axis=1)
    
    
    #batch_px_last, batch_pz_last, _ = robot_pos_model(batch_linear_vel, batch_angular_vel)
    #batch_px_last, batch_pz_last, _ = robot_pos_model(batch_linear_vel, batch_angular_vel)
    
    wandb_list = []

    #pred_horizon = batch_action_label.shape[1]
    #action_dim = batch_action_label.shape[2]

    # split into batches
    #batch_obs_images_list = torch.split(batch_obs_images, max_batch_size, dim=0)
    #batch_goal_images_list = torch.split(batch_goal_images, max_batch_size, dim=0)
    """
    uc_actions_list = []
    gc_actions_list = []
    gc_distances_list = []

    for obs, goal in zip(batch_obs_images_list, batch_goal_images_list):
        model_output_dict = model_output(
            ema_model,
            noise_scheduler,
            obs,
            goal,
            pred_horizon,
            action_dim,
            num_samples,
            device,
        )
        uc_actions_list.append(to_numpy(model_output_dict['uc_actions']))
        gc_actions_list.append(to_numpy(model_output_dict['gc_actions']))
        gc_distances_list.append(to_numpy(model_output_dict['gc_distance']))

    # concatenate
    uc_actions_list = np.concatenate(uc_actions_list, axis=0)
    gc_actions_list = np.concatenate(gc_actions_list, axis=0)
    gc_distances_list = np.concatenate(gc_distances_list, axis=0)

    # split into actions per observation
    uc_actions_list = np.split(uc_actions_list, num_images_log, axis=0)
    gc_actions_list = np.split(gc_actions_list, num_images_log, axis=0)
    gc_distances_list = np.split(gc_distances_list, num_images_log, axis=0)

    gc_distances_avg = [np.mean(dist) for dist in gc_distances_list]
    gc_distances_std = [np.std(dist) for dist in gc_distances_list]

    assert len(uc_actions_list) == len(gc_actions_list) == num_images_log

    np_distance_labels = to_numpy(batch_distance_labels)
    """
    for i in range(num_images_log):
        #fig, ax = plt.subplots(1, 3)
        fig = plt.figure(figsize=(34, 16), dpi=80)
        gs = fig.add_gridspec(2,3)
        ax_graph = fig.add_subplot(gs[0:2, 0:1])
        ax_ob = fig.add_subplot(gs[0:1, 1:2])
        ax_goal = fig.add_subplot(gs[0:1, 2:3])
        ax_inst = fig.add_subplot(gs[1:2, 1:3])
                    
        x_seq = to_numpy(batch_px_list[i])
        z_seq = to_numpy(batch_pz_list[i])
        #x_last = to_numpy(batch_px_last[i])
        #z_last = to_numpy(batch_pz_last[i])
                
        xgt = to_numpy(obj_poses[i,0])
        ygt = to_numpy(obj_poses[i,1])

        xest = to_numpy(last_poses[i,0])
        yest = to_numpy(last_poses[i,1])
        
        ax_graph.plot(x_seq, z_seq, marker = 'o', color='blue')
        ax_graph.plot(xgt, ygt, marker = '*', color='red')
        ax_graph.plot(xest, yest, marker = '+', color='green')
        #ax[0].plot(x_last, z_last, marker = 's', color='black')
                
        obs_image = to_numpy(batch_viz_obs_images[i])
        prompt = obj_inst[i]
        #print(batch_viz_obs_images[i].max(), batch_viz_obs_images[i].min(), batch_viz_obs_images[i].type())
        goal_image = to_numpy(batch_viz_goal_images[i])
        # move channel to last dimension
        obs_image = np.moveaxis(obs_image, 0, -1)
        goal_image = np.moveaxis(goal_image, 0, -1)
        ax_ob.imshow(obs_image)
        ax_goal.imshow(goal_image)
        ax_inst.text(0, 0, prompt, fontsize = 12, color = 'black')
        ax_inst.axis('off')
                        
        # set title
        ax_graph.set_title(f"est. trajectory")
        ax_ob.set_title(f"observation")
        ax_goal.set_title(f"cropped goal image")
        #ax[2].set_title(f"goal: label={np_distance_labels[i]} gc_dist={gc_distances_avg[i]:.2f}±{gc_distances_std[i]:.2f}")
        
        # make the plot large
        fig.set_size_inches(18.5, 10.5)
        
        save_path = os.path.join(visualize_path, f"sample_{i}.png")
        plt.savefig(save_path)
        wandb_list.append(wandb.Image(save_path))
        plt.close(fig)
            
        """
        fig, ax = plt.subplots(1, 3)

        x_seq = to_numpy(batch_px_list[i])
        z_seq = to_numpy(batch_pz_list[i])
        #x_last = to_numpy(batch_px_last[i])
        #z_last = to_numpy(batch_pz_last[i])
                
        xgt = to_numpy(obj_poses[i,0])
        ygt = to_numpy(obj_poses[i,1])

        xest = to_numpy(last_poses[i,0])
        yest = to_numpy(last_poses[i,1])
        
        ax[0].plot(x_seq, z_seq, marker = 'o', color='blue')
        ax[0].plot(xgt, ygt, marker = '*', color='red')
        ax[0].plot(xest, yest, marker = '+', color='green')
        #ax[0].plot(x_last, z_last, marker = 's', color='black')
                
        obs_image = to_numpy(batch_viz_obs_images[i])
        #print(batch_viz_obs_images[i].max(), batch_viz_obs_images[i].min(), batch_viz_obs_images[i].type())
        goal_image = to_numpy(batch_viz_goal_images[i])
        # move channel to last dimension
        obs_image = np.moveaxis(obs_image, 0, -1)
        goal_image = np.moveaxis(goal_image, 0, -1)
        ax[1].imshow(obs_image)
        ax[2].imshow(goal_image)

        # set title
        ax[0].set_title(f"est. trajectory")
        ax[1].set_title(f"observation")
        ax[2].set_title(f"cropped goal image")
        #ax[2].set_title(f"goal: label={np_distance_labels[i]} gc_dist={gc_distances_avg[i]:.2f}±{gc_distances_std[i]:.2f}")
        
        # make the plot large
        fig.set_size_inches(18.5, 10.5)

        save_path = os.path.join(visualize_path, f"sample_{i}.png")
        plt.savefig(save_path)
        wandb_list.append(wandb.Image(save_path))
        plt.close(fig)
        """
    if len(wandb_list) > 0 and use_wandb:
        wandb.log({f"{eval_type}_action_samples": wandb_list}, commit=False)
        
