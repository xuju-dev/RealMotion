from pathlib import Path
import numpy as np
import torch
import pytorch_lightning as pl
from hydra import compose, initialize
from hydra.utils import instantiate

from av2.map.map_api import ArgoverseStaticMap
from av2.datasets.motion_forecasting import scenario_serialization

class AV2MapVisualizer:
    """ From SIMPL """
    def __init__(self, dataset_path):
        if dataset_path is not None:
            self.dataset_dir = dataset_path
        else:
            self.dataset_dir = '/dev_ws/src/tam_deep_prediction/data/raceverse-small-v2'

    def show_map(self,
                 ax,
                 split: str,
                 seq_id: str,
                 show_freespace=True):

        # ax.set_facecolor("grey")

        static_map_path = Path(self.dataset_dir + f"/{split}/{seq_id}" + f"/log_map_archive_{seq_id}.json")
        static_map = ArgoverseStaticMap.from_json(static_map_path)

        # ~ drivable area
        # print('num drivable areas: ', len(static_map.vector_drivable_areas),
        #       [x for x in static_map.vector_drivable_areas.keys()])
        for drivable_area in static_map.vector_drivable_areas.values():
            # ax.plot(drivable_area.xyz[:, 0], drivable_area.xyz[:, 1], color='grey', alpha=0.5, linestyle='--')
            ax.fill(drivable_area.xyz[:, 0], drivable_area.xyz[:, 1], color='grey', alpha=0.2)

        # ~ lane segments
        # print('num lane segs: ', len(static_map.vector_lane_segments),
        #       [x for x in static_map.vector_lane_segments.keys()])
        print('Num lanes: ', len(static_map.vector_lane_segments))
        for lane_segment in static_map.vector_lane_segments.values():
            # print('left pts: ', lane_segment.left_lane_boundary.xyz.shape,
            #       'right pts: ', lane_segment.right_lane_boundary.xyz.shape)

            if lane_segment.lane_type == 'VEHICLE':
                lane_clr = 'blue'
            elif lane_segment.lane_type == 'BIKE':
                lane_clr = 'green'
            elif lane_segment.lane_type == 'BUS':
                lane_clr = 'orange'
            else:
                assert False, "Wrong lane type"

            # if lane_segment.is_intersection:
            #     lane_clr = 'yellow'

            polygon = lane_segment.polygon_boundary
            ax.fill(polygon[:, 0], polygon[:, 1], color=lane_clr, alpha=0.1)

            for boundary in [lane_segment.left_lane_boundary, lane_segment.right_lane_boundary]:
                ax.plot(boundary.xyz[:, 0],
                        boundary.xyz[:, 1],
                        linewidth=1,
                        color='grey',
                        alpha=0.3)

            # cl = static_map.get_lane_segment_centerline(lane_segment.id)
            # ax.plot(cl[:, 0], cl[:, 1], linestyle='--', color='magenta', alpha=0.1)

        # ~ ped xing
        for pedxing in static_map.vector_pedestrian_crossings.values():
            edge = np.concatenate([pedxing.edge1.xyz, np.flip(pedxing.edge2.xyz, axis=0)])
            # plt.plot(edge[:, 0], edge[:, 1], color='orange', alpha=0.75)
            ax.fill(edge[:, 0], edge[:, 1], color='orange', alpha=0.2)
            # for edge in [ped_xing.edge1, ped_xing.edge2]:
            #     ax.plot(edge.xyz[:, 0], edge.xyz[:, 1], color='orange', alpha=0.5, linestyle='dotted')

    def show_map_clean(self,
                       ax,
                       split: str,
                       seq_id: str,
                       show_freespace=True):

        # ax.set_facecolor("grey")

        static_map_path = Path(self.dataset_dir + f"/{split}/{seq_id}" + f"/log_map_archive_{seq_id}.json")
        static_map = ArgoverseStaticMap.from_json(static_map_path,
                                                  overwrite_centerline=False)

        # ~ drivable area
        for drivable_area in static_map.vector_drivable_areas.values():
            # ax.plot(drivable_area.xyz[:, 0], drivable_area.xyz[:, 1], color='grey', alpha=0.5, linestyle='--')
            ax.fill(drivable_area.xyz[:, 0], drivable_area.xyz[:, 1], color='grey', alpha=0.2)

        # ~ lane segments
        print('Num lanes: ', len(static_map.vector_lane_segments))
        for lane_id, lane_segment in static_map.vector_lane_segments.items():
            # lane_clr = 'grey'
            polygon = lane_segment.polygon_boundary
            ax.fill(polygon[:, 0], polygon[:, 1], color='whitesmoke', alpha=1.0, edgecolor=None, zorder=0)

            # centerline
            centerline = lane_segment.centerline.xyz[:, 0:2]  # use xy
            ax.plot(centerline[:, 0], centerline[:, 1], alpha=0.1, color='grey', linestyle='dotted', zorder=1)

            # # lane boundary
            # for boundary, mark_type in [(lane_segment.left_lane_boundary.xyz, lane_segment.left_mark_type),
            #                             (lane_segment.right_lane_boundary.xyz, lane_segment.right_mark_type)]:

            #     clr = None
            #     width = 1.0
            #     if mark_type in [LaneMarkType.DASH_SOLID_WHITE,
            #                      LaneMarkType.DASHED_WHITE,
            #                      LaneMarkType.DOUBLE_DASH_WHITE,
            #                      LaneMarkType.DOUBLE_SOLID_WHITE,
            #                      LaneMarkType.SOLID_WHITE,
            #                      LaneMarkType.SOLID_DASH_WHITE]:
            #         clr = 'white'
            #         zorder = 3
            #         width = width
            #     elif mark_type in [LaneMarkType.DASH_SOLID_YELLOW,
            #                        LaneMarkType.DASHED_YELLOW,
            #                        LaneMarkType.DOUBLE_DASH_YELLOW,
            #                        LaneMarkType.DOUBLE_SOLID_YELLOW,
            #                        LaneMarkType.SOLID_YELLOW,
            #                        LaneMarkType.SOLID_DASH_YELLOW]:
            #         clr = 'gold'
            #         zorder = 4
            #         width = width * 1.1

            #     style = 'solid'
            #     if mark_type in [LaneMarkType.DASHED_WHITE,
            #                      LaneMarkType.DASHED_YELLOW,
            #                      LaneMarkType.DOUBLE_DASH_YELLOW,
            #                      LaneMarkType.DOUBLE_DASH_WHITE]:
            #         style = (0, (5, 10))  # loosely dashed
            #     elif mark_type in [LaneMarkType.DASH_SOLID_YELLOW,
            #                        LaneMarkType.DASH_SOLID_WHITE,
            #                        LaneMarkType.DOUBLE_SOLID_YELLOW,
            #                        LaneMarkType.DOUBLE_SOLID_WHITE,
            #                        LaneMarkType.SOLID_YELLOW,
            #                        LaneMarkType.SOLID_WHITE,
            #                        LaneMarkType.SOLID_DASH_WHITE,
            #                        LaneMarkType.SOLID_DASH_YELLOW]:
            #         style = 'solid'

            #    if (clr is not None) and (style is not None):
            #        ax.plot(boundary[:, 0],
            #                boundary[:, 1],
            #                color=clr,
            #                alpha=1.0,
            #                linewidth=width,
            #                linestyle=style,
            #                zorder=zorder)

        # ~ ped xing
        for pedxing in static_map.vector_pedestrian_crossings.values():
            edge = np.concatenate([pedxing.edge1.xyz, np.flip(pedxing.edge2.xyz, axis=0)])
            ax.fill(edge[:, 0], edge[:, 1], color='yellow', alpha=0.1, edgecolor=None)

def load_model(checkpoint_path: str, split: str = "test"):
    # Load preprocessed data
    with initialize(config_path="./conf"):
        cfg = compose(config_name="config")
        cfg.checkpoint = checkpoint_path

    # Ensure reproducibility
    pl.seed_everything(cfg.seed)

    # Load datamodule
    test = True if split == 'test' else False
    datamodule = instantiate(cfg.datamodule.pl_module, test=test)
    datamodule.setup(stage=split)

    # Load model
    model = instantiate(cfg.model.pl_module)
    checkpoint = torch.load(cfg.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])

    # Prep dataloader
    loader = None
    if split == 'train':
        loader = datamodule.train_dataloader()
    elif split == 'val':
        loader = datamodule.val_dataloader()
    elif split == 'test':
        loader = datamodule.test_dataloader()

    return loader, model

def load_scenario_and_map(scenario_id: str, split: str, dataset_root: str) -> tuple:
    """
    Load scenario data and static map from a given scenario ID and split.

    Args:
        scenario_id (str): The scenario ID (e.g., '001-182cf688453e7cb4-11-abudhabi')
        split (str): The dataset split (e.g., 'train', 'val', 'test')
        dataset_root (str): Root path to the dataset (e.g., '/path/to/raceverse-small-v2')

    Returns:
        Tuple containing:
            - scenario (Scenario): Loaded scenario object
            - static_map (ArgoverseStaticMap): Loaded static map object
    """
    scenario_path = Path(dataset_root) / split / scenario_id

    # Find parquet file
    parquet_files = list(scenario_path.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {scenario_path}")
    parquet_path = parquet_files[0]
    # print(f"[INFO] Found scenario parquet at: {parquet_path}")

    # Map path
    static_map_path = scenario_path / f"log_map_archive_{scenario_id}.json"
    if not static_map_path.exists():
        raise FileNotFoundError(f"Map JSON not found: {static_map_path}")

    # Load
    scenario = scenario_serialization.load_argoverse_scenario_parquet(parquet_path)
    static_map = ArgoverseStaticMap.from_json(static_map_path)

    return scenario, static_map

def local_to_global(traj_local, center, angle=0):
    """Convert [T, 2] trajectory from local to global using rotation and translation."""
    R = torch.tensor([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ], dtype=traj_local.dtype, device=traj_local.device)

    return traj_local @ R.T + center

def extract_predicted_traj(preds, agent_idx: int, batch_idx: int = 0, all_agents: bool = True):
    """
    Extracts predicted trajectories and returns them as polylines.

    Args:
        preds: Dictionary containing model predictions.
        batch_idx: Index of the batch to extract from.
        agent_idx: Index of the agent to extract from.
        all_agents: True plots all agents (`range(agent_idx)`), False plots only agent_idx
    Returns:
        List of polylines, each polyline is a numpy array of shape [T, 2].
    """
    glo_y_hat = preds['memory_dict']['glo_y_hat']  # [B, A, T, 2]
    B, A, T, _ = glo_y_hat.shape
    if batch_idx > B or agent_idx > A:
        raise IndexError(f"Batch index {batch_idx} or agent index {agent_idx} out of range: B={B}, A={A}")

    centers = preds['memory_dict']['origin']  # [B, 2]
    predicted_trajectories = []

    if all_agents:
        # print("Collecting all agents...")
        for a in range(A):
            center = centers[batch_idx]
            traj = glo_y_hat[batch_idx, a]  # [T, 2]
            polyline = local_to_global(traj, center)
            polyline = polyline.cpu().numpy()  # Convert to NumPy
            predicted_trajectories.append(polyline)     # [T, 2]
    else:
        # print(f"Collecting agent {agent_idx}...")
        center = centers[batch_idx]
        traj = glo_y_hat[batch_idx, agent_idx]  # [T, 2]
        polyline = local_to_global(traj, center)
        polyline = polyline.cpu().numpy()  # Convert to NumPy
        predicted_trajectories.append(polyline)     # [T, 2]

    return predicted_trajectories

def extract_targets(labels, batch_idx: int, agent_idx: int, all_agents: bool):
    target = labels['target']  # [B, A, 60, 2]
    t_mask = labels['target_mask']  # [B, A, 60]
    masked_target = target * t_mask.unsqueeze(-1)  # shape: [B, A, 60, 2]

    B, A, T, _ = target.shape
    if batch_idx > B or agent_idx > A:
        raise IndexError(f"Batch index {batch_idx} or agent index {agent_idx} out of range: B={B}, A={A}")

    centers = labels['origin']
    angles = labels['theta']

    trajectories = []
    if all_agents:
        # print("Collecting all agents...")
        for a in range(A):
            traj = masked_target[batch_idx, a]  # [T, 2]
            # polyline = local_to_global(traj, centers[batch_idx], angles[batch_idx])
            polyline = traj.cpu().numpy()  # Convert to NumPy
            trajectories.append(polyline)     # [T, 2]
    else:
        # print(f"Collecting agent {agent_idx}...")
        traj = masked_target[batch_idx, agent_idx]  # [T, 2]
        polyline = local_to_global(traj, centers[batch_idx], angles[batch_idx])
        polyline = polyline.cpu().numpy()  # Convert to NumPy
        trajectories.append(polyline)     # [T, 2]

    return trajectories
