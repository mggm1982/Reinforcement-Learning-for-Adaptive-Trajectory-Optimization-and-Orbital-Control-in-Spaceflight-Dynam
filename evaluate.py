import numpy as np
from stable_baselines3 import PPO
from envs.orbit_env import CowellOrbitEnv

def evaluate_model(model_path="models/latest_model.zip", steps=2000, render_every=100):
    """
    Evaluates a PPO model trained with differential reward.
    Logs distance, velocity difference, and reward progression.
    """

    print(f"🧠 Loading model: {model_path}")
    model = PPO.load(model_path)
    env = CowellOrbitEnv()
    obs, _ = env.reset()

    trajectory = {
        "step": [],
        "distance": [],
        "velocity_diff": [],
        "reward": [],
        "mass": [],
        "dist_reduction": [],
        "energy_reduction": [],
        "prograde_align": [],
    }

    total_reward = 0.0
    for step in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        trajectory["step"].append(step)
        trajectory["distance"].append(info["distance_to_target"])
        trajectory["velocity_diff"].append(info["velocity_diff"])
        trajectory["reward"].append(reward)
        trajectory["mass"].append(info["mass"])
        trajectory["dist_reduction"].append(info["dist_reduction"])
        trajectory["energy_reduction"].append(info["energy_reduction"])
        trajectory["prograde_align"].append(info["prograde_align"])

        if step % render_every == 0:
            print(
                f"Step {step:4d}: "
                f"dist={info['distance_to_target']:.1f} m, "
                f"vel_diff={info['velocity_diff']:.2f} m/s, "
                f"reward={reward:+.4f}"
            )

        if terminated or truncated:
            print(f"\n✅ Episode ended at step {step}.")
            break

    np.savez("eval_results.npz", **trajectory)
    print(f"\n✅ Evaluation data saved to eval_results.npz")
    print(f"Total cumulative reward: {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    evaluate_model(model_path="safe_meta_rl_final.zip", steps=2000, render_every=100)
