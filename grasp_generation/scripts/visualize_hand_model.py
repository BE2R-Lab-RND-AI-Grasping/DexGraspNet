"""
Last modified date: 2022.03.11
Author: mzhmxzh
Description: visualize hand model
"""

import os
import sys

os.chdir(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.realpath('.'))

import numpy as np
import torch
import trimesh as tm
import transforms3d
import plotly.graph_objects as go
from utils.hand_model import HandModel


torch.manual_seed(1)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    device = torch.device('cpu')

    # hand model

    hand_model = HandModel(
        mjcf_path='mjcf/shadow_hand_wrist_free.xml',
        mesh_path='mjcf/meshes',
        contact_points_path='mjcf/contact_points.json',
        penetration_points_path='mjcf/penetration_points.json',
        device=device
    )
    
    rot = transforms3d.euler.euler2mat(-np.pi / 2, -np.pi / 2, 0, axes='rzyz')
    hand_pose = torch.cat([
        torch.tensor([0, 0, 0], dtype=torch.float, device=device), 
        # torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float, device=device),
        torch.tensor(rot.T.ravel()[:6], dtype=torch.float, device=device),
        # torch.zeros([16], dtype=torch.float, device=device),
        torch.tensor([
            0, 0.5, 0, 0, 
            0, 0.5, 0, 0, 
            0, 0.5, 0, 0, 
            1.4, 0, 0, 0, 
        ], dtype=torch.float, device=device), 
    ], dim=0)

    hand_pose = torch.tensor([[-0.2219,  0.0148, -0.0169,  0.2890,  0.8152,  0.5019, -0.6498, -0.2180,
          0.7282, -0.2136,  0.3764,  0.8096,  0.2025, -0.1860,  0.4484,  0.9593,
          0.1439, -0.0475,  0.1477,  0.5907,  0.4202,  0.1601, -0.1513,  0.3288,
          0.7487,  0.1484,  0.2015,  1.0665,  0.1229, -0.2054, -0.0341]])
    hand_model.set_parameters(hand_pose)

    # info
    contact_candidates = hand_model.get_contact_candidates()[0]
    surface_points = hand_model.get_surface_points()[0]
    print(f'n_dofs: {hand_model.n_dofs}')
    print(f'n_contact_candidates: {len(contact_candidates)}')
    print(f'n_surface_points: {len(surface_points)}')
    print(hand_model.chain.get_joint_parameter_names())

    # visualize

    hand_plotly = hand_model.get_plotly_data(i=0, opacity=0.5, color='lightblue')
    v = contact_candidates.detach().cpu()
    contact_candidates_plotly = [go.Scatter3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], mode='markers', marker=dict(size=2, color='green'))]
    v = surface_points.detach().cpu()
    surface_points_plotly = [go.Scatter3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], mode='markers', marker=dict(size=2, color='lightblue'))]
    
    fig = go.Figure(hand_plotly + contact_candidates_plotly + surface_points_plotly)
    fig.update_layout(scene_aspectmode='data')
    fig.show()