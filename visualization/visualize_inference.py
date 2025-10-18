import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from vis_utils import AV2MapVisualizer, load_scenario_and_map
from av2.datasets.motion_forecasting.viz.scenario_visualization import _plot_actor_tracks


infernce_file_name = "single_agent_2025-10-15-00-28"
inference_file_path = f"/dev_ws/src/tam_deep_prediction/models/RealMotion/RealMotion/submission/{infernce_file_name}.parquet"

checkpoint_path = "/dev_ws/src/tam_deep_prediction/models/RealMotion/RealMotion/outputs/RealMotion_I-av2/20251012-021826/outputs/RealMotion_I-av2/20251012-021826/checkpoints/epoch_79-minADE6_0.9717246890068054.ckpt"

dataset_path = "/dev_ws/src/tam_deep_prediction/data/tum_av2_heading_90"  # path to the original dataset
split = 'test'

viz_output_dir = f"/dev_ws/src/tam_deep_prediction/models/RealMotion/RealMotion/outputs/visualizations/inference/{infernce_file_name}"
os.makedirs(viz_output_dir, exist_ok=True)

top_k = 3
t_hist = 20
t_fut = 40

# Read inference results
preds_df = pd.read_parquet(inference_file_path)

scenario_ids = preds_df['scenario_id'].unique()

for scenario_id in scenario_ids:
    # print(f"Visualizing scenario {scenario_id}...")
    scenario_preds = preds_df[preds_df['scenario_id'] == scenario_id]
    best_prob = scenario_preds['probability'].max()
    
    sorted_preds = sorted(scenario_preds['probability'], reverse=True)  
    top_k_preds = sorted_preds[:top_k]

    # Some plot configs
    _, ax = plt.subplots(figsize=(8, 8))
    ax.axis('equal')
    ax.set_title('{}-{}'.format(scenario_id, 'motion-forecasting'))

    scenario, static_map = load_scenario_and_map(scenario_id, split, dataset_path)
    focal_id = scenario.focal_track_id
    preds_focal = scenario_preds[scenario_preds['track_id'] == focal_id]
    
    ### Visualizing map and agents ###
    AV2MapVisualizer(dataset_path=dataset_path).show_map(ax, split=split, seq_id=scenario_id)
    _plot_actor_tracks(ax, scenario, timestep=20)  # only history

    ### Plot ground truth ###
    focal_track = next(t for t in scenario.tracks if t.track_id == scenario.focal_track_id)
    positions = np.array([state.position for state in focal_track.object_states])
    gt_future = positions[t_hist : t_hist + t_fut]
    gt_fut2end = positions[t_hist + t_fut: ]
    final_gt_pos = gt_future[-1, -1]

    ax.plot(gt_future[:,0], gt_future[:,1], 'r--', label='Ground Truth Future')

    ### Plot predictions ###
    # Plot text box with best probability
    ax.text(
        0.98, 0.98,
        f"Best Probability: {best_prob:.2f}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(
            boxstyle="square,pad=0.4",
            fc="white", 
            ec="black", 
            lw=1
        )
    )

    # ax.plot(gt_fut2end[:,0], gt_fut2end[:,1], 'g--', label='Ground Truth End')
    # Get the last observed position of the focal agent
    last_observed_pos = positions[t_hist - 1]
    print("Last observed:", positions[t_hist-1])
    first_pred_x = preds_focal['predicted_trajectory_x'].values[0][0]
    first_pred_y = preds_focal['predicted_trajectory_y'].values[0][0]
    
    print("First predicted:", first_pred_x, first_pred_y)
    
    # Check agent speed to estimate time offset
    speed = np.linalg.norm(positions[t_hist-1] - positions[t_hist-2]) / 0.1  # m/s
    offset_m = np.linalg.norm(gt_future[0] - [first_pred_x, first_pred_y])
    print("Approx time offset (s):", offset_m / speed)


    # Plot top k predicted trajectories for the target agent
    for prob in top_k_preds:
        preds = scenario_preds[scenario_preds['probability'] == prob]
        preds_x = preds['predicted_trajectory_x'].values[0]
        preds_y = preds['predicted_trajectory_y'].values[0]
        
        ax.plot(preds_x, preds_y, linestyle='--', color='blue', linewidth=1)

        # last_observed_pos = positions[t_hist - 1]
        # print("Last observed:", last_observed_pos)
        # print("First predicted:", preds_x[0], preds_y[0])

        # Add probability text at the end of trajectory
        ax.text(
            preds_x[-1],
            preds_y[-1],
            f"{prob:.2f}",
            fontsize=7,
            color='blue',
            verticalalignment='bottom',
            horizontalalignment='right'
        )

    ax.legend()
    viz_save_path = Path(viz_output_dir) / f"new2_{scenario_id}inference.png"
    plt.savefig(viz_save_path, dpi=100, bbox_inches="tight")
    print(f"Saved visualization to {viz_save_path}")
