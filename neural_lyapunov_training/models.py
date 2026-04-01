# Definitions of dynamics (inverted pendulum), lyapunov function
# and loss function.
# Everything needs to be a subclass of nn.Module in order to be handled by
# auto_LiRPA.

import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
import torch
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append("../torchdyn")

import neural_lyapunov_training.lyapunov as lyapunov
import neural_lyapunov_training.controllers as controllers
import neural_lyapunov_training.dynamical_system as dynamical_system
import neural_lyapunov_training.pendulum as pendulum

def create_model(
    dynamics,
    controller_parameters=None,
    lyapunov_parameters=None,
    loss_parameters=None,
    path=None,
    lyapunov_func="lyapunov.NeuralNetworkLyapunov",
    loss_func="lyapunov.LyapunovDerivativeLoss",
    controller_func="controllers.NeuralNetworkController",
):
    """
    Build the computational graph for verification of general dynamics + controller + neural lyapunov function.
    """
    # Default parameters.
    if controller_parameters is None:
        controller_parameters = {
            "nlayer": 2,
            "hidden_dim": 5,
            "clip_output": None,
        }
    if lyapunov_parameters is None:
        lyapunov_parameters = {
            # 'nlayer': 3,
            "hidden_widths": [32, 32],
            "R_rows": 0,
            "absolute_output": False,
            "eps": 0.0,
            "activation": nn.ReLU,
        }
    if loss_parameters is None:
        loss_parameters = {
            "kappa": 0.1,
        }
    controller = eval(controller_func)(
        in_dim=dynamics.x_equilibrium.size(0),
        out_dim=dynamics.u_equilibrium.size(0),
        x_equilibrium=dynamics.x_equilibrium,
        u_equilibrium=dynamics.u_equilibrium,
        **controller_parameters,
    )
    lyapunov_nn = eval(lyapunov_func)(
        x_dim=dynamics.x_equilibrium.size(0),
        goal_state=dynamics.x_equilibrium,
        **lyapunov_parameters,
    )

    loss = eval(loss_func)(dynamics, controller, lyapunov_nn, **loss_parameters)
    # TODO: load a trained model. Currently using random weights, just to test autoLiRPA works.
    if path is not None:
        loss.load_state_dict(torch.load(path))
    return loss


def create_output_feedback_model(
    dynamics,
    controller_parameters,
    lyapunov_parameters,
    path=None,
    observer_parameters=None,
    loss_parameters=None,
    lyapunov_func="lyapunov.NeuralNetworkLyapunov",
    loss_func="lyapunov.LyapunovDerivativeDOFLoss",
    controller_func="controllers.NeuralNetworkController",
    observer_func="controllers.NeuralNetworkLuenbergerObserver",
):
    """
    Build the computational graph for verification of general dynamics + controller + neural lyapunov function.
    """
    if loss_parameters is None:
        loss_parameters = {
            "kappa": 0,
        }
    nx = dynamics.continuous_time_system.nx
    ny = dynamics.continuous_time_system.ny
    nu = dynamics.continuous_time_system.nu
    h = lambda x: dynamics.continuous_time_system.h(x)
    controller = eval(controller_func)(
        in_dim=nx + ny,
        out_dim=nu,
        x_equilibrium=torch.concat((dynamics.x_equilibrium, torch.zeros(ny))),
        u_equilibrium=dynamics.u_equilibrium,
        **controller_parameters,
    )
    lyapunov_nn = eval(lyapunov_func)(
        x_dim=2 * nx,
        goal_state=torch.concat((dynamics.x_equilibrium, torch.zeros(nx))),
        **lyapunov_parameters,
    )
    observer = eval(observer_func)(
        nx, ny, dynamics, h, torch.zeros(1, ny), observer_parameters["fc_hidden_dim"]
    )
    loss = eval(loss_func)(
        dynamics, observer, controller, lyapunov_nn, **loss_parameters
    )
    if path is not None:
        loss.load_state_dict(torch.load(path)["state_dict"])
    return loss


def create_pendulum_model_state_feedback(**kwargs):
    return create_model(
        dynamical_system.SecondOrderDiscreteTimeSystem(
            pendulum.PendulumDynamics(m=0.15, l=0.5, beta=0.1),
            dt=0.05,
            position_integration=dynamical_system.IntegrationMethod.ExplicitEuler,
            velocity_integration=dynamical_system.IntegrationMethod.ExplicitEuler,
        ),
        **kwargs,
    )


def create_pendulum_model(dt=0.01, **kwargs):
    """
    Build the computational graph for verification of the inverted pendulum model.
    """
    # Create the "model" (the entire computational graph for computing Lyapunov loss). Make sure all parameters here match colab.
    return create_model(
        dynamical_system.SecondOrderDiscreteTimeSystem(
            pendulum.PendulumDynamics(m=0.15, l=0.5, beta=0.1), dt
        ),
        **kwargs,
    )


def create_pendulum_output_feedback_model(dt=0.01, **kwargs):
    """
    Build the computational graph for verification of the inverted pendulum model.
    """
    # Create the "model" (the entire computational graph for computing Lyapunov loss). Make sure all parameters here match colab.
    return create_output_feedback_model(
        dynamical_system.SecondOrderDiscreteTimeSystem(
            pendulum.PendulumDynamics(m=0.15, l=0.5, beta=0.1), dt
        ),
        **kwargs,
    )




def add_hole(box_low, box_high, inner_low, inner_high):
    boxes_low = []
    boxes_high = []
    for i in range(box_low.size(0)):
        # Split on dimension i.
        box1_low = box_low.clone()
        box1_low[i] = inner_high[i]
        box1_high = box_high.clone()
        box2_low = box_low.clone()
        box2_high = box_high.clone()
        box2_high[i] = inner_low[i]
        boxes_low.extend([box1_low, box2_low])
        boxes_high.extend([box1_high, box2_high])
        box_low[i] = inner_low[i]
        box_high[i] = inner_high[i]
    boxes_low = torch.stack(boxes_low, dim=0)
    boxes_high = torch.stack(boxes_high, dim=0)
    return boxes_low, boxes_high


def box_data(
    eps=None, lower_limit=-1.0, upper_limit=1.0, ndim=2, scale=1.0, hole_size=0
):
    """
    Generate a box between (-1, -1) and (1, 1) as our region to verify stability.
    We may place a small hole around the origin.
    """
    if isinstance(lower_limit, list):
        data_min = scale * torch.tensor(
            lower_limit, dtype=torch.get_default_dtype()
        ).unsqueeze(0)
    else:
        data_min = scale * torch.ones((1, ndim)) * lower_limit
    if isinstance(upper_limit, list):
        data_max = scale * torch.tensor(
            upper_limit, dtype=torch.get_default_dtype()
        ).unsqueeze(0)
    else:
        data_max = scale * torch.ones((1, ndim)) * upper_limit
    if hole_size != 0:
        inner_low = data_min.squeeze(0) * hole_size
        inner_high = data_max.squeeze(0) * hole_size
        data_min, data_max = add_hole(
            data_min.squeeze(0), data_max.squeeze(0), inner_low, inner_high
        )
    X = (data_min + data_max) / 2.0
    # Assume the "label" is 1, so we verify the positiveness.
    labels = torch.ones(size=(data_min.size(0),), dtype=torch.int64)
    # Lp norm perturbation epsilon. Not used, since we will return per-element min and max.
    eps = None
    return X, labels, data_max, data_min, eps


def simulate(lyaloss: lyapunov.LyapunovDerivativeLoss, steps: int, x0):
    # Assumes explicit euler integration.
    x_traj = [None] * steps
    V_traj = [None] * steps
    x_traj[0] = x0
    with torch.no_grad():
        V_traj[0] = lyaloss.lyapunov.forward(x_traj[0])
        for i in range(1, steps):
            u = lyaloss.controller.forward(x_traj[i - 1])
            x_traj[i] = lyaloss.dynamics.forward(x_traj[i - 1], u)
            V_traj[i] = lyaloss.lyapunov.forward(x_traj[i])

    return x_traj, V_traj
