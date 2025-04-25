
# Handover Optimization in 5G Networks Using Deep Q-Learning

**Course:** CS462 - 5G Communication and Network Laboratory  
**Faculty:** Dr. Bhupendra Kumar  
**Teaching Assistants:** Shivam Solanki & Abhinav Jain

---

## 📘 Project Overview

In 5G networks, efficient handover management is crucial for maintaining seamless connectivity and high Quality of Service (QoS). Traditional handover algorithms often rely on static thresholds, which may not adapt well to dynamic network conditions.  
This project leverages **reinforcement learning**, specifically **Deep Q-Learning (DQL)**, to enable adaptive and intelligent handover decisions.

The agent interacts with a custom environment simulating user equipment (UE) movement and network conditions. Through training, the agent learns optimal policies to decide when and where to perform handovers, reducing ping-pong effects and improving overall network efficiency.

---


## Presentation Video

You can watch our project presentation video [here](https://www.youtube.com/watch?v=mxbdsCFHodU). Below are the key timestamps:

- **00:00** - Introduction
- **05:56** - Environment Setup
- **14:10** - Algorithm Explained
- **32:07** - Conclusion with Results

---

## 📂 Project Structure

- `main.py`: Entry point for training and evaluating the DQL agent. Handles configuration, environment setup, and orchestrates the learning process.
- `environment.py`: Defines the simulation environment, including state representation, reward calculation, and transition dynamics for the handover scenario.
- `rl_algo.py`: Implements the Deep Q-Learning algorithm, including the neural network model, experience replay, and training logic.
- `requirements.txt`: Lists all Python dependencies required to run the project.

---

## Project Setup

### 1. Install Dependencies

Make sure you have Python 3.8+ installed. Install required packages using:

```bash
pip install -r requirements.txt
```

### 2. Run the Project

To start training the agent:

```bash
python main.py
```

You can modify hyperparameters and environment settings in `main.py` as needed.

---

## Key Features

- Custom 5G handover simulation environment (`environment.py`)
- Deep Q-Network (DQN) agent with experience replay and target network (`rl_algo.py`)
- Configurable training and evaluation pipeline (`main.py`)
- Easily extensible for further research and experimentation

---


## Contributors

- **Harshit Agrawal (202251053)**
- **Malaika Varshney (202251069)**
- **Dharmik Vyas (202251039)**
- **Abhyudaya Tiwari (202251004)**

---