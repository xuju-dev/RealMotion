# Visualize map and agents given a scenario_id. For debugging purposes.

import matplotlib.pyplot as plt
import os
from pathlib import Path

from vis_utils import (
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

dataset_path = '/dev_ws/src/tam_deep_prediction/data/raceverse-small-v2/'
split = 'train'
# scenario_id = '004-182cfa1c1f757318-8'
# scenario_id = '005-182d15857c8abd03-2'
# after new preprocessing
# scenario_id = '004-182cf9ffc29a6a05-12'
scenario_id = '004-182cfa37039d7682-6'

scenario, static_map = load_scenario_and_map(scenario_id, split, dataset_path)
print(f"Loaded scenario {scenario_id}.")

# Some plot configs
_, ax = plt.subplots(figsize=(8, 8))
ax.axis('equal')
subtitle = 'map_and_agents'
ax.set_title('{}-{}'.format(scenario_id, subtitle))

# Visualizing map
AV2MapVisualizer(dataset_path=dataset_path).show_map(ax, split=split, seq_id=scenario_id)

_plot_actor_tracks(ax, scenario, timestep=40)

ax.legend()
viz_output_dir = Path("/dev_ws/src/tam_deep_prediction/models/RealMotion/RealMotion/visualizations/debug")  # path where the figure gets saved to
if not os.path.exists(viz_output_dir):
    print("Path for Visualization does not exist. Creating directory...")
    os.mkdir(viz_output_dir)

viz_save_path = viz_output_dir / f"{scenario_id}_{subtitle}.png"

plt.savefig(viz_save_path)
print(f"Saved visualization to {viz_save_path}\n")
