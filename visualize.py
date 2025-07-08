import os
from pathlib import Path
import torch

from models.RealMotion.RealMotion.vis_utils import (
    AV2MapVisualizer,
    load_model,
    load_scenario_and_map,
    local_to_global,
    extract_predicted_traj
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
        y_hat = preds['y_hat'][sample_idx, agent_idx]  # [60, 2]
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
split = 'test'
loader, model = load_model(checkpoint_path=CHECKPOINT_PATH, split=split)

features, labels, extra = next(iter(loader))

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model.to(device)

features = {k: v.to(device) if torch.is_tensor(v) else v for k, v in features.items()}
labels = {k: v.to(device) if torch.is_tensor(v) else v for k, v in labels.items()}
extra = {k: v.to(device) if torch.is_tensor(v) else v for k, v in extra.items()}

scenario_ids = features['scenario_id']

# === MAP VISUALIZATION===
sample_idx = 0
agent_idx = 4
scenario_id = scenario_ids[sample_idx]

# Load scenario for visualization
print(f"Loading scenario {scenario_id}...")
scenario, static_map = load_scenario_and_map(scenario_id, split, dataset_path)

_, ax = plt.subplots(figsize=(12, 12))
ax.axis('equal')
ax.set_title('{}-{}-agent{}'.format(scenario_id, task, agent_idx))
ax.legend()

# Visualizing map
print("Visualizing map...")
AV2MapVisualizer(dataset_path=dataset_path).show_map(ax, split=split, seq_id=scenario_id)

print("Plotting actor tracks...")
_plot_actor_tracks(ax, scenario, timestep=60)

# === PREDICT ===
if MODE == 1:
    model.eval()
    with torch.no_grad():
        print("Predicting...\n")
        preds = model(features)

# === TRAJETORY VISUALIZATION ===
if preds is not None:
    predicted_trajectories = extract_predicted_traj(preds, batch_idx=sample_idx, all_agents=False, agent_idx=agent_idx)

    # Plot polylines
    print("Plotting predicted trajectories...\n")
    _plot_polylines(predicted_trajectories)

    # Plot past trajectory
    # print("Visualizing trajectories...\n")
    # # for agent in range(y_hat.shape[1]):
    # for agent in range(1):
    #     print(f"Plotting agent {agent}...")
    #     plot_trajectories_on_map(ax, features, preds=preds, sample_idx=sample_idx, agent_idx=agent)

# === SAVE VISUALIZATION ===
viz_output_dir = Path(PROJECT_ROOT, "visualizations")  # path where the figure gets saved to
if not os.path.exists(viz_output_dir):
    print("Path for Visualization does not exist. Creating directory...")
    os.mkdir(viz_output_dir)

viz_save_path = viz_output_dir / f"{scenario_id}_{agent_idx}_{task}.png"

print(f"Saving visualization to {viz_save_path}...")
plt.savefig(viz_save_path)
