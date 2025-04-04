# Kivy Self-Driving Car Simulation with TD3

## Overview

This project implements a 2D self-driving car simulation using the Kivy framework for the graphical interface and a Twin Delayed Deep Deterministic Policy Gradient (TD3) agent for the car's control logic. The agent learns to navigate a predefined path marked on a map, aiming to reach a sequence of target goal points.

## Features

* **2D Car Simulation:** Built with the Python Kivy framework.
* **Path Following:** The environment uses a `mask.png` image to define the valid path (black lines on a white background).
* **Goal-Oriented Navigation:** The car attempts to navigate through a predefined sequence of goal points (A1, A2, A3...).
* **Reinforcement Learning Agent:** Uses the TD3 algorithm, an advanced actor-critic method suitable for continuous control problems.
* **Sensor-Based State:** The AI agent receives input from:
    * Three forward-facing sensors detecting the path.
    * Orientation relative to the current goal.
    * Proximity to the window boundaries.
* **Model Persistence:** Allows saving and loading the trained TD3 agent (Actor and Critic networks).
* **Basic UI:** Includes buttons for saving/loading models and potentially clearing (resetting simulation state).

## Algorithm: TD3 (Twin Delayed Deep Deterministic Policy Gradient)

This project utilizes TD3, a state-of-the-art model-free, off-policy actor-critic algorithm designed primarily for continuous action spaces.

### Benefits of Using TD3 Here:

1.  **Handles Continuous Actions:** TD3 is inherently designed for continuous action spaces. In this simulation, the action is the car's rotation angle. TD3 can output fine-grained continuous values for this rotation, allowing for potentially smoother and more precise steering control compared to discretizing the action space (as required by algorithms like DQN).
2.  **Improved Stability:** Compared to its predecessor DDPG, TD3 incorporates several techniques to enhance training stability and performance:
    * **Twin Critics:** Learns two independent Q-value (Critic) networks and uses the minimum of their predictions to calculate the target value. This helps reduce overestimation bias, leading to more reliable value estimates.
    * **Delayed Policy Updates:** The policy (Actor) network is updated less frequently than the critic networks. This allows the value estimates to stabilize somewhat before being used to update the policy, preventing chasing noisy targets.
    * **Target Policy Smoothing:** Adds clipped noise to the target action during the target Q calculation. This smooths the value estimate and makes the policy less likely to exploit sharp peaks in the Q-function.
3.  **Sample Efficiency:** As an off-policy algorithm, TD3 uses a Replay Buffer to store past experiences (state, action, reward, next\_state, done). It samples batches from this buffer to perform updates, allowing it to reuse experiences multiple times and often leading to better sample efficiency compared to on-policy methods that discard experience after one update.

## Setup and Installation

1.  **Clone/Download:** Get the project files.
2.  **Create Environment (Recommended):**
    ```bash
    python -m venv venv## Running the Simulation

1.  Navigate to the project directory in your terminal.
2.  Ensure your Python environment (e.g., `venv`) is activated.
3.  Run the main script:
    ```bash
    python map.py
    ```
4.  The Kivy window will appear. The car will perform random actions for the first `start_timesteps` (default: 1000) before using the TD3 policy.
5.  Training happens automatically during the simulation based on the `train_freq` parameter.
6.  Use the "save" and "load" buttons to manage the trained agent's state. A reward plot is saved to `./results/` when you save the model.

## Configuration & Tuning

Key parameters can be adjusted near the top of `map.py`:

* `state_dim`, `action_dim`, `max_action`: Define the AI interface.
* `start_timesteps`: Duration of initial random exploration.
* `expl_noise`: Scale of noise added to actions for exploration.
* `train_freq`: How many environment steps between training calls.
* Reward values within `Game.update`: The balance of rewards/penalties significantly impacts learning.
* `base_speed` within `Car.move`: Affects simulation speed and control dynamics.
* TD3 hyperparameters (discount, tau, policy\_noise, etc.) can be adjusted within `td3_agent.py` or passed during `policy.train` if needed.
