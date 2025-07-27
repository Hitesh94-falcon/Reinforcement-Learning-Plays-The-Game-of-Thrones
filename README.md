# 🐉 RL Plays Game of Thrones — Q-learning Environment

This project is a **Reinforcement Learning (RL)** simulation inspired by the famous Tv serial **Game of Thrones**. An agent (Daenerys Targaryen) navigates a 6×6 grid, captures 3 kingdoms, collects dragon eggs, avoids enemies (Dragons), and reaches the Iron Throne using **Q-learning**.

---

## 📌 Project Structure

- `Q_learning.py`: Q-learning training logic  
- `my_env.py`: Custom Gymnasium environment  
- `train.py`: Main training runner  
- `test.py`: Evaluation/testing script  
- `utils.py`: Logging utilities  
- `Images/`: Game sprites (agent, goal, obstacles, etc.)  
  - `obstacles/`
  - `rewards/`
  - `states/`
- `q_tables/`: Saved Q-tables (.npy files)  
- `logs/`: Training logs (CSV)  
- `requirements.txt`: Required Python packages  
- `Padm Presentation.pptx`: Project presentation


## 🚀 Features

- Multi-phase Q-learning with separate Q-tables:
- Kingdom 1 → Kingdom 2 → Kingdom 3 → Goal
- Dynamic reward shaping
- Pygame rendering
- Visualization of Q-tables using seaborn heatmaps
- Logging of rewards, epsilon, and success/failure per episode

## 🛠️ How to Run

### 📦 1. Install Dependencies

```bash
1. pip install -r requirements.txt

2. python train.py #set visualization to true to visualize the Q-tables

3. python test.py


Acknowledgements
Project developed for Principles of Autonomy and Decision Making
Inspired by: OpenAI Gym, Game of Thrones, MinimalRL

Future Improvements
Implement Deep Q-Learning (DQN)
Build user interface for dynamic level design

Author
Hitesh Suresh
M.Sc. Artificial Intelligence for Autonomous Systems
Technische Hochschule Ingolstadt