# 2D Drone Navigation with PPO

This project was developed as part of the university module **"Einführung in neuronale Netzwerke für Studierende der Physik"** (Introduction to Neural Networks for Physics Students). The project was carried out as a **group effort by three students**.

## Project Overview
The objective of this project is to train a drone to navigate within a simulated 2D environment using Reinforcement Learning. Specifically, the drone must learn to fly from a random starting point to a random target location while handling varying initial conditions (position, velocity, and orientation). 

The drone is controlled by adjusting the thrust of its **left and right rotors**, requiring the agent to learn complex stabilization and navigation maneuvers simultaneously.

## How it Works
The project implements the **Proximal Policy Optimization (PPO)** algorithm, an Actor-Critic method. The workflow includes:
1. **Data Collection:** The drone interacts with the environment, storing trajectories (states, actions, rewards).
2. **Advantage Estimation:** Using the Critic's evaluations, the system calculates the "Advantage" to determine which actions led to better-than-expected outcomes.
3. **Optimization:** Both networks are updated using a clipped loss function to ensure stable and reliable policy improvements.

## Neural Network Architecture
The architecture consists of two separate neural networks that interact during the training process:

### 1. The Actor (Policy Network)
The Actor is responsible for choosing the actions.
* **Input:** 8-dimensional state vector (x, y, $v_x$, $v_y$, angle $\phi$, angular velocity $\omega$, and distance to target $dist_x$, $dist_y$).
* **Hidden Layer:** 64 neurons with **ReLU** activation.
* **Output:** 4 neurons representing the **Mean ($\mu$)** and **Standard Deviation ($\sigma$)** for the two rotors.
* **Action Logic:** Actions are sampled from a normal distribution. During the interaction, a **tanh** function is applied to the sampled values to keep the rotor thrust within a normalized range of [-1, 1].

### 2. The Critic (Value Network)
The Critic evaluates the current state of the environment.
* **Input:** 8-dimensional state vector.
* **Hidden Layer:** 128 neurons with **ReLU** activation.
* **Output:** 1 neuron (linear) that estimates the expected future cumulative reward (State Value $V(s)$).

## Technical Details
* **Framework:** TensorFlow / Keras
* **Environment:** Custom 2D Drone Environment (Gymnasium-based)
* **Training:** The model is typically trained over **2,000 iterations** to achieve stable and precise flight control.
