# ============================================================
# REINFORCEMENT LEARNING - Smart Irrigation (Q-Learning)
# State: soil moisture level (0-10) | Actions: water 0-3 units
# Reward: negative distance from optimal moisture (target=6)
# ============================================================
import os
import numpy as np

os.makedirs("models", exist_ok=True)

# ---------- Environment ----------
class IrrigationEnv:
    TARGET = 6  # optimal soil moisture

    def reset(self):
        self.moisture = np.random.randint(1, 9)   # random start 1-8
        return self.moisture

    def step(self, action):
        self.moisture = min(10, self.moisture + action)      # agent waters
        self.moisture = max(0, self.moisture -
                            np.random.choice([0, 1], p=[0.3, 0.7]))  # evaporation
        reward = -abs(self.moisture - self.TARGET)           # closer = better
        return self.moisture, reward

# ---------- Q-Learning ----------
env = IrrigationEnv()
Q = np.zeros((11, 4))          # 11 moisture states x 4 actions

ALPHA, GAMMA = 0.1, 0.9        # learning rate, discount factor
EPSILON, EPS_MIN, DECAY = 1.0, 0.05, 0.995
EPISODES, STEPS = 5000, 50

reward_history = []
for ep in range(EPISODES):
    s = env.reset()
    total_reward = 0
    for _ in range(STEPS):
        # epsilon-greedy action selection
        if np.random.rand() < EPSILON:
            a = np.random.randint(4)                 # explore
        else:
            a = np.argmax(Q[s])                      # exploit
        s2, r = env.step(a)
        # Q-value update (Bellman equation)
        Q[s, a] += ALPHA * (r + GAMMA * np.max(Q[s2]) - Q[s, a])
        s = s2
        total_reward += r
    EPSILON = max(EPS_MIN, EPSILON * DECAY)
    reward_history.append(total_reward)

# ---------- Results ----------
print("Learned policy (best action per moisture level 0-10):")
print(np.argmax(Q, axis=1))

avg_last = np.mean(reward_history[-500:])
print(f"\nAverage reward (last 500 episodes): {avg_last:.1f}")
print("(early episodes average:", f"{np.mean(reward_history[:500]):.1f})")

# ---------- Plot reward curve ----------
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.plot(reward_history, alpha=0.3, color="gray")
plt.plot(np.convolve(reward_history, np.ones(100)/100, mode="valid"), color="red")
plt.xlabel("Episode"); plt.ylabel("Total Reward")
plt.title("Q-Learning Reward Curve (agent improves over time)")
plt.savefig("rl_reward.png", dpi=120, bbox_inches="tight")

# ---------- Save ----------
np.save("models/irrigation_q.npy", Q)
print("✅ Saved -> models/irrigation_q.npy")