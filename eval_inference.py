import os

import numpy as np
import pandas as pd

# ========= 1. Load parquet files =========
pred_df = pd.read_parquet("/dev_ws/src/tam_deep_prediction/models/RealMotion/RealMotion/submission/single_agent_2025-10-15-00-28.parquet")
dataset_path = "/dev_ws/src/tam_deep_prediction/data/tum_av2_heading_90"  # path to the original dataset
split = 'test'

# ========= 2. Metric functions =========
def ADE(pred, gt):
    """Average Displacement Error"""
    return np.linalg.norm(pred - gt, axis=1).mean()

def FDE(pred, gt):
    """Final Displacement Error"""
    return np.linalg.norm(pred[-1] - gt[-1])

# ========= 3. Metric computation =========
minADE1_list, minADE6_list = [], []
minFDE1_list, minFDE6_list = [], []
miss_rate_list = []

MISS_THRESHOLD = 2.0  # meters (standard)

for scenario_id, group in pred_df.groupby("scenario_id"):
    # Get track_id to filter GT
    track_id = group.iloc[0]["track_id"]

    # Get gt scenario file path
    gt_path = os.path.join(dataset_path, split, scenario_id, f'scenario_{scenario_id}.parquet')
    print(gt_path)
    if not os.path.exists(gt_path):
        print(f"[WARN] Missing GT for scenario {scenario_id}, skipping.")
        continue

    # Load GT parquet for this scenario
    gt_df = pd.read_parquet(gt_path)
    # Get the trajectory of the specific track_id
    gt = gt_df[gt_df["track_id"] == track_id].sort_values("timestep")[20:60][["position_x", "position_y"]].to_numpy()
    history = gt_df[gt_df["track_id"] == track_id].sort_values("timestep")[0:20][["position_x", "position_y"]].to_numpy()

    # Sort predictions by probability and get trajectories
    sorted_group = group.sort_values("probability", ascending=False)
    preds = [
        np.stack([np.array(row["predicted_trajectory_x"]), np.array(row["predicted_trajectory_y"])], axis=-1)
        for _, row in sorted_group.iterrows()
    ]

    # Debug
    print('### Debug ###')
    print('History:\n', history[-1])
    print('Preds:\n', preds[:3])
    print('GT:\n', gt[:3])
    # limit to top 6 modes
    top6 = preds[:6]
    top1 = preds[0]

    # compute ADE/FDE for each trajectory
    ades = [ADE(p, gt) for p in top6]
    fdes = [FDE(p, gt) for p in top6]

    # ---- minADE1 / minFDE1 ----
    minADE1_list.append(ades[0])
    minFDE1_list.append(fdes[0])

    # ---- minADE6 / minFDE6 ----
    minADE6_list.append(min(ades))
    minFDE6_list.append(min(fdes))

    # ---- MR6 ---- (miss if all FDEs > threshold)
    miss_rate_list.append(all(f > MISS_THRESHOLD for f in fdes))

# ========= 4. Aggregate metrics =========
metrics = {
    "minADE1": np.mean(minADE1_list),
    "minADE6": np.mean(minADE6_list),
    "minFDE1": np.mean(minFDE1_list),
    "minFDE6": np.mean(minFDE6_list),
    "MR": np.mean(miss_rate_list),
}

# ========= 5. Print results =========
print("\n=== RealMotion_I Evaluation Metrics ===")
for k, v in metrics.items():
    print(f"{k:8s}: {v:.4f}")
