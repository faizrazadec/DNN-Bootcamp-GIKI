# Space Adventure AI: A Neural Network-Powered Journey 🚀

Welcome to **Space Adventure AI**, a thrilling Flappy Bird-style game where you navigate a spaceship through challenging obstacles. What makes this project unique is the integration of an AI that learns to control the spaceship, enhancing its decision-making abilities as it flies through space!

---

## 🎮 Project Overview

**Space Adventure AI** is more than just a game; it’s an exploration of how neural networks can be applied to real-time gaming scenarios. This project was developed during the **AI Bootcamp** at **Ghulam Ishaq Khan Institute of Engineering Sciences and Technology (GIKI)**, sponsored by **SkyElectric Pvt. Ltd.**, under the guidance of **[Sri Usama](https://www.linkedin.com/in/usamajanjua9/)**.

---

### Key Features:
- **AI-Controlled Gameplay**: The AI model is trained to play the game, making decisions based on the current state of the game environment.
- **Real-Time Data Collection**: Training data is collected during gameplay, allowing the AI to improve its performance over time.
- **Dynamic Obstacles**: The game features randomly generated obstacles that keep the gameplay challenging and engaging.
- **Space-Themed Visuals**: Immersive visuals with space-themed graphics and a user-friendly interface.

---

## 🧠 AI Model

The AI model is built using **TensorFlow** and consists of a simple neural network trained to control the spaceship. The model is trained on data collected during gameplay, including the position of the spaceship, speed, and distance from obstacles.

### Model Architecture:
- **Input Layer**: 8 features representing the game state.
- **Hidden Layers**: 2 Dense layers with ReLU activation.
- **Output Layer**: A single neuron with a sigmoid activation function, representing the probability of performing a jump.

---

## 💻 Technologies Used

- **Python**: The primary programming language used for development.
- **Pygame**: Used for game development, rendering, and managing game events.
- **TensorFlow**: For building and training the AI model.

---

## 🚀 How to Run the Project

### Prerequisites:
- Python 3.x installed on your system.
- Required Python libraries: `pygame`, `tensorflow`, `numpy`, `pickle`.

### Installation:
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/space-adventure-ai.git
