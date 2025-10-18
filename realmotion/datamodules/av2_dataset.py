from typing import List
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class Av2Dataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: str = None,
        num_historical_steps: int = 20,
        split_points: List[int] = [50],
        radius: float = 300.0,
        logger=None,
    ):
        assert num_historical_steps <= 20
        assert all(p >= num_historical_steps for p in split_points), \
            "Split points must be >= num_historical_steps."
        assert split in ['train', 'val', 'test']
        super(Av2Dataset, self).__init__()
        self.data_folder = Path(data_root) / split
        self.file_list = sorted(list(self.data_folder.glob('*.pt')))
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = 0 if split == 'test' else 40
        self.split_points = split_points
        self.radius = radius

        if logger is not None:
            logger.info(f'data root: {data_root}/{split}, total number of files: {len(self.file_list)}')

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: int):
        try:
            data = torch.load(self.file_list[index])
            data = self.process(data)
            return data
        except Exception as e:
            print(f"Failed to load {self.file_list[index]}: {e}")
            raise

    def process(self, data):
        sequence_data = []
        for cur_step in self.split_points:
            ag_dict = self.process_single_agent(data, cur_step)
            sequence_data.append(ag_dict)
        return sequence_data

    def process_single_agent(self, data, step=50):
        idx = data['focal_idx']
        cur_agent_id = data['agent_ids'][idx]
        origin = data['x_positions'][idx, step - 1]
        theta = data['x_angles'][idx, step - 1]
        rotate_mat = torch.tensor(
            [
                [torch.cos(theta), -torch.sin(theta)],
                [torch.sin(theta), torch.cos(theta)],
            ],
        )
        ag_mask = torch.norm(data['x_positions'][:, step - 1] - origin, dim=-1) < self.radius
        ag_mask = ag_mask * data['x_valid_mask'][:, step - 1]
        ag_mask[idx] = False

        # transform agents to local
        st, ed = step - self.num_historical_steps, step + self.num_future_steps
        attr = torch.cat([data['x_attr'][[idx]], data['x_attr'][ag_mask]])
        pos = data['x_positions'][:, st: ed]
        pos = torch.cat([pos[[idx]], pos[ag_mask]])
        head = data['x_angles'][:, st: ed]
        head = torch.cat([head[[idx]], head[ag_mask]])
        vel = data['x_velocity'][:, st: ed]
        vel = torch.cat([vel[[idx]], vel[ag_mask]])
        valid_mask = data['x_valid_mask'][:, st: ed]
        valid_mask = torch.cat([valid_mask[[idx]], valid_mask[ag_mask]])

        pos[valid_mask] = torch.matmul(pos[valid_mask] - origin, rotate_mat)
        head[valid_mask] = (head[valid_mask] - theta + np.pi) % (2 * np.pi) - np.pi

        # transform lanes to local
        l_pos = data['lane_positions']
        l_attr = data['lane_attr']
        l_is_int = data['is_intersections']
        l_pos = torch.matmul(l_pos.reshape(-1, 2) - origin, rotate_mat).reshape(-1, l_pos.size(1), 2)

        l_ctr = l_pos[:, 9:11].mean(dim=1)
        l_head = torch.atan2(
            l_pos[:, 10, 1] - l_pos[:, 9, 1],
            l_pos[:, 10, 0] - l_pos[:, 9, 0],
        )
        l_valid_mask = (
            (l_pos[:, :, 0] > -self.radius) & (l_pos[:, :, 0] < self.radius)
            & (l_pos[:, :, 1] > -self.radius) & (l_pos[:, :, 1] < self.radius)
        )

        l_mask = l_valid_mask.any(dim=-1)
        l_pos = l_pos[l_mask]
        l_is_int = l_is_int[l_mask]
        l_attr = l_attr[l_mask]
        l_ctr = l_ctr[l_mask]
        l_head = l_head[l_mask]
        l_valid_mask = l_valid_mask[l_mask]

        l_pos = torch.where(
            l_valid_mask[..., None], l_pos, torch.zeros_like(l_pos)
        )

        # remove outliers
        nearest_dist = torch.cdist(pos[:, self.num_historical_steps - 1, :2],
                                   l_pos.view(-1, 2)).min(dim=1).values
        ag_mask = nearest_dist < 5
        ag_mask[0] = True
        pos = pos[ag_mask]
        head = head[ag_mask]
        vel = vel[ag_mask]
        attr = attr[ag_mask]
        valid_mask = valid_mask[ag_mask]

        # post_process
        head = head[:, :self.num_historical_steps]
        vel = vel[:, :self.num_historical_steps]
        pos_ctr = pos[:, self.num_historical_steps - 1].clone()
        if self.num_future_steps > 0:
            type_mask = attr[:, [-1]] != 3
            pos, target = pos[:, :self.num_historical_steps], pos[:, self.num_historical_steps:]
            target_mask = type_mask & valid_mask[:, [self.num_historical_steps - 1]] & valid_mask[:, self.num_historical_steps:]
            valid_mask = valid_mask[:, :self.num_historical_steps]
            target = torch.where(
                target_mask.unsqueeze(-1),
                target - pos_ctr.unsqueeze(1), torch.zeros_like(target),   
            )
        else:
            target = target_mask = None

        diff_mask = valid_mask[:, :self.num_historical_steps - 1] & valid_mask[:, 1: self.num_historical_steps]
        tmp_pos = pos.clone()
        pos_diff = pos[:, 1:self.num_historical_steps] - pos[:, :self.num_historical_steps - 1]
        pos[:, 1:self.num_historical_steps] = torch.where(
            diff_mask.unsqueeze(-1),
            pos_diff, torch.zeros(pos.size(0), self.num_historical_steps - 1, 2)
        )
        pos[:, 0] = torch.zeros(pos.size(0), 2)

        tmp_vel = vel.clone()
        vel_diff = vel[:, 1:self.num_historical_steps] - vel[:, :self.num_historical_steps - 1]
        vel[:, 1:self.num_historical_steps] = torch.where(
            diff_mask,
            vel_diff, torch.zeros(vel.size(0), self.num_historical_steps - 1)
        )
        vel[:, 0] = torch.zeros(vel.size(0))

        return {
            'target': target,
            'target_mask': target_mask,
            'x_positions_diff': pos,
            'x_positions': tmp_pos,
            'x_attr': attr,
            'x_centers': pos_ctr,
            'x_angles': head,
            'x_velocity': tmp_vel,
            'x_velocity_diff': vel,
            'x_valid_mask': valid_mask,
            'lane_positions': l_pos,
            'lane_centers': l_ctr,
            'lane_angles': l_head,
            'lane_attr': l_attr,
            'lane_valid_mask': l_valid_mask,
            'is_intersections': l_is_int,
            'origin': origin.view(1, 2),
            'theta': theta.view(1),
            'scenario_id': data['scenario_id'],
            'track_id': cur_agent_id,
            'city': data['city'],
            'timestamp': torch.Tensor([step * 0.1])
        }


def collate_fn(seq_batch):
    seq_data = []
    for i in range(len(seq_batch[0])):
        batch = [b[i] for b in seq_batch]
        data = {}

        for key in [
            'x_positions_diff',
            'x_attr',
            'x_positions',
            'x_centers',
            'x_angles',
            'x_velocity',
            'x_velocity_diff',
            'lane_positions',
            'lane_centers',
            'lane_angles',
            'lane_attr',
            'is_intersections',
        ]:
            data[key] = pad_sequence([b[key] for b in batch], batch_first=True)

        if 'x_scored' in batch[0]:
            data['x_scored'] = pad_sequence(
                [b['x_scored'] for b in batch], batch_first=True
            )

        if batch[0]['target'] is not None:
            data['target'] = pad_sequence([b['target'] for b in batch], batch_first=True)
            data['target_mask'] = pad_sequence(
                [b['target_mask'] for b in batch], batch_first=True, padding_value=False
            )

        for key in ['x_valid_mask', 'lane_valid_mask']:
            data[key] = pad_sequence(
                [b[key] for b in batch], batch_first=True, padding_value=False
            )

        data['x_key_valid_mask'] = data['x_valid_mask'].any(-1)
        data['lane_key_valid_mask'] = data['lane_valid_mask'].any(-1)

        data['scenario_id'] = [b['scenario_id'] for b in batch]
        data['track_id'] = [b['track_id'] for b in batch]

        data['origin'] = torch.cat([b['origin'] for b in batch], dim=0)
        data['theta'] = torch.cat([b['theta'] for b in batch])
        data['timestamp'] = torch.cat([b['timestamp'] for b in batch])
        seq_data.append(data)
    return seq_data


# def collate_fn(seq_batch):
#     seq_data = []

#     for i in range(len(seq_batch[0])):
#         batch = [b[i] for b in seq_batch]
#         data = {}

#         # Max # of agents in this batch
#         max_agents = max(b['x_positions'].shape[0] for b in batch)

#         def pad_agents(tensor, target_len):
#             if tensor.shape[0] == target_len:
#                 return tensor
#             pad_shape = list(tensor.shape)
#             pad_shape[0] = target_len - tensor.shape[0]
#             pad = torch.zeros(*pad_shape, dtype=tensor.dtype, device=tensor.device)
#             return torch.cat([tensor, pad], dim=0)

#         def pad_agents_bool(tensor, target_len):
#             if tensor.shape[0] == target_len:
#                 return tensor
#             pad_shape = list(tensor.shape)
#             pad_shape[0] = target_len - tensor.shape[0]
#             pad = torch.zeros(*pad_shape, dtype=torch.bool, device=tensor.device)
#             return torch.cat([tensor, pad], dim=0)

#         # Agent-level tensors
#         agent_keys = [
#             'x_positions_diff', 'x_attr', 'x_positions', 'x_centers',
#             'x_angles', 'x_velocity', 'x_velocity_diff'
#         ]
#         for key in agent_keys:
#             data[key] = pad_sequence([b[key] for b in batch], batch_first=True)

#         # Lane-level (only 1 lane set per batch element)
#         lane_keys = ['lane_positions', 'lane_centers', 'lane_angles', 'lane_attr']
#         for key in lane_keys:
#             data[key] = pad_sequence([b[key] for b in batch], batch_first=True)

#         # Booleans (masks)
#         data['x_valid_mask'] = torch.stack([
#             pad_agents_bool(b['x_valid_mask'], max_agents) for b in batch
#         ])
#         data['x_key_valid_mask'] = data['x_valid_mask'].any(-1)

#         data['lane_valid_mask'] = pad_sequence(
#             [b['lane_valid_mask'] for b in batch],
#             batch_first=True,
#             padding_value=False
#         )
#         data['lane_key_valid_mask'] = data['lane_valid_mask'].any(-1)

#         # Optional fields
#         if 'x_scored' in batch[0]:
#             data['x_scored'] = torch.stack([
#                 pad_agents_bool(b['x_scored'], max_agents) for b in batch
#             ])
#         if batch[0]['target'] is not None:
#             data['target'] = torch.stack([
#                 pad_agents(b['target'], max_agents) for b in batch
#             ])
#             data['target_mask'] = torch.stack([
#                 pad_agents_bool(b['target_mask'], max_agents) for b in batch
#             ])

#         # Metadata
#         data['scenario_id'] = [b['scenario_id'] for b in batch]
#         data['track_id'] = [b['track_id'] for b in batch]
#         data['origin'] = torch.stack([b['origin'] for b in batch])
#         data['theta'] = torch.stack([b['theta'] for b in batch])
#         data['timestamp'] = torch.stack([b['timestamp'] for b in batch])

#         # Optional: Agent validity mask (for padding tracking): True for true agents, False for padding
#         agent_mask = torch.zeros(len(batch), max_agents, dtype=torch.bool)
#         for j, b in enumerate(batch):
#             agent_mask[j, :b['x_positions'].shape[0]] = True
#         data['agent_mask'] = agent_mask

#         seq_data.append(data)

#     return seq_data
