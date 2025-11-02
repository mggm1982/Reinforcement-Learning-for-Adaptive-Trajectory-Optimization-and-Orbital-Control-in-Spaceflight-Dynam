import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
from envs.orbit_env import CowellOrbitEnv
from stable_baselines3 import PPO


def visualize_trajectory(model_path="models/latest_model.zip", steps=200000):
    """
    Visualize the trained orbit transfer trajectory with:
      - Earth surface sphere
      - Initial and target orbit rings
      - Actual trajectory path
      - Ideal Hohmann transfer ellipse overlay
    """

    print("🌍 Generating trajectory visualization...")

    # Environment + model
    env = CowellOrbitEnv()
    model = PPO.load(model_path)
    obs, _ = env.reset()

    positions = []
    rewards = []

    for step in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        positions.append(env.state[:3].numpy())
        rewards.append(reward)

        if terminated or truncated:
            print(f"✅ Episode ended at step {step}")
            break

    positions = np.array(positions)

    # --- Earth + orbit rings ---
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Earth surface (blue sphere)
    earth_r = 6.371e6
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    x = earth_r * np.cos(u) * np.sin(v)
    y = earth_r * np.sin(u) * np.sin(v)
    z = earth_r * np.cos(v)
    ax.plot_surface(x, y, z, color='royalblue', alpha=0.5)

    # Initial & target orbit radii
    r_init = 6.7e6
    r_target = 7.2e6
    theta = np.linspace(0, 2 * np.pi, 400)
    x_init = r_init * np.cos(theta)
    y_init = r_init * np.sin(theta)
    x_target = r_target * np.cos(theta)
    y_target = r_target * np.sin(theta)

    ax.plot(x_init, y_init, 0, 'g--', label="Initial Orbit")
    ax.plot(x_target, y_target, 0, 'r--', label="Target Orbit")

    # Actual trajectory
    xs, ys, zs = positions[:, 0], positions[:, 1], positions[:, 2]
    c = np.linspace(0, 1, len(xs))
    ax.plot(xs, ys, zs, color='gold', linewidth=2.5, label="Agent Trajectory")

    # --- Ideal Hohmann Transfer Ellipse ---
    mu = env.mu
    a = 0.5 * (r_init + r_target)  # semi-major axis
    e = (r_target - r_init) / (r_target + r_init)  # eccentricity

    # True anomaly range (0 to π for half-ellipse)
    theta_transfer = np.linspace(0, np.pi, 300)
    r_hoh = (a * (1 - e ** 2)) / (1 + e * np.cos(theta_transfer))
    x_hoh = r_hoh * np.cos(theta_transfer)
    y_hoh = r_hoh * np.sin(theta_transfer)
    ax.plot(x_hoh, y_hoh, 0, 'c-', linewidth=2.0, label="Ideal Hohmann Transfer")

    # Mark the perigee burn point (where prograde burn should start)
    ax.scatter(r_init, 0, 0, color='orange', s=60, label="Hohmann Burn Point")

    # --- Visual styling ---
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("Orbit Transfer Trajectory Visualization", fontsize=14)
    ax.legend(loc="upper right")
    ax.set_box_aspect([1, 1, 1])
    ax.grid(True)

    # Adjust limits dynamically based on orbit size
    max_r = 1.1 * r_target
    for axis in [ax.set_xlim, ax.set_ylim, ax.set_zlim]:
        axis(-max_r, max_r)

    plt.tight_layout()
    plt.savefig("orbit_visualization_hohmann.png", dpi=200)
    plt.show()

    print("✅ Visualization complete with Earth, orbits, and Hohmann overlay.")
    env.close()


if __name__ == "__main__":
    visualize_trajectory(model_path="safe_meta_rl_final.zip", steps=2000000)
