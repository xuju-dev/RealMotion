import torch
from pathlib import Path

# Path to your preprocessed data (adjust as needed)
data_root = Path("/dev_ws/src/tam_deep_prediction/data/raceverse-small-v2/realmotion_processed_incl_unscored/train")
# scene_file = data_root / "scenario_004-182cf9ffc29a6a05-12.pt"
scene_file = data_root / "scenario_004-182cfa37039d7682-6.pt"

# Load the dictionary
data = torch.load(scene_file)

# Now you can inspect it
for k in data:
    print(f"{k}: {type(data[k])}, shape: {getattr(data[k], 'shape', 'N/A')}")

print(f"Scenario: {data['scenario_id']}")
print(f"Number of agents: {len(data['agent_ids'])}")
print(f"x_positions shape: {data['x_positions'].shape}")
print(f"x_valid_mask shape: {data['x_valid_mask'].shape}")

valid_agents_mask = data["x_valid_mask"].any(dim=1)
num_valid_agents = valid_agents_mask.sum().item()
print(f"Number of valid agents: {num_valid_agents}")

print("Agent info:")
for i, agent_id in enumerate(data["agent_ids"]):
    valid = valid_agents_mask[i].item()
    obj_type = data['x_attr'][i, 0].item()
    obj_cat = data['x_attr'][i, 1].item()
    print(f"  Agent {i} (ID: {agent_id}): valid={valid}, type={obj_type}, category={obj_cat}")