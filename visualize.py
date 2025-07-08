import os
from pathlib import Path
import torch

from models.RealMotion.RealMotion.vis_utils import (
    AV2MapVisualizer,
    load_model,
    load_scenario_and_map,
    local_to_global,
    extract_predicted_traj,
    extract_targets
)

from av2.datasets.motion_forecasting.viz.scenario_visualization import (
    _plot_actor_tracks,
    _plot_polylines
)

import numpy as np
import matplotlib.pyplot as plt

def plot_trajectories_on_map(
    ax,
    features,
    preds=None,
    sample_idx=0,
    agent_idx=0,
    title="Trajectories on Map"
):
    """
    Plot agent trajectories on top of the lane map.

    Args:
        ax: matplotlib axis where the lane map is plotted on
        features: dict of batched features
        preds: prediction dict (optional, from model output)
        sample_idx: index in batch to plot
        agent_idx: which agent (0–5)
        plot_pred: whether to show model prediction
        title: plot title
    """
    # === Extract agent info ===
    x_pos = features['x_positions'][sample_idx, agent_idx]     # [30, 2]
    x_mask = features['x_valid_mask'][sample_idx, agent_idx]   # [30]

    center = preds['memory_dict']['origin'][sample_idx]
    angle = preds['memory_dict']['theta'][-1]  # (last angle)

    # Convert past to global
    past_traj_local = x_pos[x_mask.bool()]    # [T1, 2]
    past_traj = local_to_global(past_traj_local, center, angle).cpu().numpy()
    ax.plot(past_traj[:, 0], past_traj[:, 1], color='blue', label='Past')

    # Prediction
    if preds is None:
        raise ValueError("Predictions are required to plot future trajectories.")
    else:
        # y_hat = preds['y_hat'][sample_idx, agent_idx]  # [60, 2]
        # pred_traj = local_to_global(y_hat, center, angle).cpu().numpy()
        global_y_hat = preds['memory_dict']['glo_y_hat'][sample_idx, agent_idx]  # [60, 2]
        global_y_hat = local_to_global(global_y_hat, center)
        pred_traj = global_y_hat.cpu().numpy()
        ax.plot(pred_traj[:, 0], pred_traj[:, 1], 'r--', label=f'Predicted Agent {agent_idx}')

# === CONFIGURATION ===
dataset_path = "/dev_ws/src/tam_deep_prediction/data/raceverse-small-v2"  # path to original dataset

# need to use absolute paths to work
PROJECT_ROOT = Path(__file__).resolve().parent  # /home/xuju/tam_deep_prediction/models/RealMotion/RealMotion
# TAM_DEEP_PREDICTION_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# CONFIG_PATH = os.path.join(PROJECT_ROOT, "conf/model/RealMotion.yaml")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "outputs/RealMotion-av2_3frame_30his/20250609-185705/checkpoints/epoch_71-minADE6_1.1839549541473389.ckpt")
epoch = 71

MODE = 1  # TODO: for naming convention (0: only map or 1: with trajectory)
if MODE == 0:
    task = "map"
elif MODE == 1:
    task = "predict"

# === LOAD MODEL ===
split = 'val'
loader, model = load_model(checkpoint_path=CHECKPOINT_PATH, split=split)

features, labels, extra = next(iter(loader))

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model.to(device)

features = {k: v.to(device) if torch.is_tensor(v) else v for k, v in features.items()}
labels = {k: v.to(device) if torch.is_tensor(v) else v for k, v in labels.items()}
extra = {k: v.to(device) if torch.is_tensor(v) else v for k, v in extra.items()}

scenario_ids = features['scenario_id']

B, A, T, _ = features['x_positions'].shape

# === PREDICT ===
if MODE == 1:
    model.eval()
    with torch.no_grad():
        print("PREDICTING...\n")
        preds = model(features)
centers = features['origin']  # [B, 2]
angles = features['theta']  # [B]

# === MAP VISUALIZATION===
# sample_idx = 3
# agent_idx = 5
B = 16
all_agents_bool = False
for sample_idx in range(B):
    print(f"### BATCH {sample_idx} ###\n")
    for agent_idx in range(A):
        # print(f"Plotting batch {sample_idx}, agent {agent_idx}...\n")
        scenario_id = scenario_ids[sample_idx]

        # Load scenario for visualization
        scenario, static_map = load_scenario_and_map(scenario_id, split, dataset_path)
        # print(f"Loaded scenario {scenario_id}.")

        # Some plot configs
        _, ax = plt.subplots(figsize=(8, 8))
        ax.axis('equal')
        ax.set_title('{}-{}-agent{}'.format(scenario_id, task, agent_idx))

        # Visualizing map
        AV2MapVisualizer(dataset_path=dataset_path).show_map(ax, split=split, seq_id=scenario_id)
        # print("Visualizing map: DONE\n")

        _plot_actor_tracks(ax, scenario, timestep=60)
        # print("Plotting actor tracks: DONE\n")

        if preds is not None:
            # === TARGET VISUALIZATION ===
            target_trajectories = extract_targets(labels, batch_idx=sample_idx, all_agents=all_agents_bool, agent_idx=agent_idx)
            _plot_polylines(target_trajectories, line_width=1, color='green')
            # Give plotted preds a label
            pred_lines = plt.gca().lines[-1]
            pred_lines.set_label('Target')
            # print("Plotting target: DONE\n")

            # === TRAJETORY VISUALIZATION ===
            predicted_trajectories = extract_predicted_traj(preds, batch_idx=sample_idx, all_agents=all_agents_bool, agent_idx=agent_idx)

            # Plot polylines
            _plot_polylines(predicted_trajectories, line_width=0.5)
            # Give plotted preds a label
            pred_lines = plt.gca().lines[-1]
            pred_lines.set_label('Prediction')
            # print("Plotting predicted trajectories: DONE\n")

        if all_agents_bool is False:
            # === SAVE VISUALIZATION ===
            ax.legend()
            viz_output_dir = Path(PROJECT_ROOT, "visualizations")  # path where the figure gets saved to
            if not os.path.exists(viz_output_dir):
                print("Path for Visualization does not exist. Creating directory...")
                os.mkdir(viz_output_dir)

            viz_save_path = viz_output_dir / f"{scenario_id}_{agent_idx}_{task}.png"
            plt.savefig(viz_save_path)
            print(f"Saved visualization to {viz_save_path}\n")
        # ---End of agents loop

    if all_agents_bool is True:
        # === SAVE VISUALIZATION ===
        ax.legend()
        viz_output_dir = Path(PROJECT_ROOT, "visualizations")  # path where the figure gets saved to
        if not os.path.exists(viz_output_dir):
            print("Path for Visualization does not exist. Creating directory...")
            os.mkdir(viz_output_dir)

        viz_save_path = viz_output_dir / f"{scenario_id}_{task}.png"

        plt.savefig(viz_save_path)
        print(f"Saved visualization to {viz_save_path}\n")
