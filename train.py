import yaml
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from envs.orbit_env import CowellOrbitEnv
import numpy as np
import copy
import os

def make_env():
    """Factory function for SubprocVecEnv compatibility."""
    def _init():
        return CowellOrbitEnv(dt=2.0, max_steps=5000)  # Slightly longer per episode
    return _init


def meta_train(meta_epochs=10, base_inner_steps=300_000):
    # Load hyperparameters
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    print("🚀 Starting Extended Safe Meta-RL PPO Training ...")
    print(f"⚙️ Config: {cfg}")

    # Adjust rollout length dynamically for better stability
    n_envs = int(cfg.get("n_envs", 4))
    env = SubprocVecEnv([make_env() for _ in range(n_envs)])

    meta_model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=float(cfg.get("learning_rate", 3e-4)),
        gamma=float(cfg.get("gamma", 0.995)),             # Longer horizon
        batch_size=int(cfg.get("batch_size", 1024)),
        ent_coef=float(cfg.get("ent_coef", 0.001)),       # Lower entropy for stability
        n_steps=8192,                                     # More rollout steps per update
        n_epochs=20,                                      # More gradient updates
        clip_range=0.2,
        tensorboard_log="./tb_logs_long/",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Meta-RL task list
    tasks = [
        {"name": "orbit_transfer"},
        {"name": "drag_perturb"},
    ]

    os.makedirs("checkpoints", exist_ok=True)

    # --- Meta training loop ---
    for epoch in range(meta_epochs):
        print(f"\n🧩 Meta-Epoch {epoch+1}/{meta_epochs}")

        # Dynamic step scheduling — longer training in later epochs
        inner_steps = base_inner_steps + epoch * 50_000

        adapted_models = []
        for task in tasks:
            print(f"   Adapting to task: {task['name']} (steps={inner_steps})")

            env_task = SubprocVecEnv([make_env() for _ in range(n_envs)])
            model_task = PPO(
                "MlpPolicy",
                env_task,
                verbose=0,
                learning_rate=float(cfg.get("learning_rate", 3e-4)),
                gamma=float(cfg.get("gamma", 0.995)),
                batch_size=int(cfg.get("batch_size", 1024)),
                ent_coef=float(cfg.get("ent_coef", 0.001)),
                n_steps=8192,
                n_epochs=20,
                clip_range=0.2,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

            # Copy base model → task model
            model_task.policy.load_state_dict(copy.deepcopy(meta_model.policy.state_dict()))

            # Inner-loop fine-tuning
            model_task.learn(total_timesteps=inner_steps)
            adapted_models.append(model_task)

        # Meta-update (average weights across tasks)
        with torch.no_grad():
            for p_meta, *p_tasks in zip(
                meta_model.policy.parameters(),
                *[m.policy.parameters() for m in adapted_models]
            ):
                p_meta.copy_(torch.mean(torch.stack([p.data for p in p_tasks]), dim=0))

        # Save checkpoints
        model_path = f"checkpoints/meta_epoch_{epoch+1}.zip"
        meta_model.save(model_path)
        print(f"💾 Saved checkpoint: {model_path}")
        print(f"✅ Meta epoch {epoch+1}/{meta_epochs} complete.")

    # Final save
    meta_model.save("safe_meta_rl_final.zip")
    print("\n🏁 Meta-training complete.")
    print("✅ Final model saved: safe_meta_rl_final.zip")

    return meta_model


def main():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    meta_epochs = int(cfg.get("meta_epochs", 10))
    inner_steps = int(cfg.get("inner_steps", 300_000))

    meta_train(meta_epochs=meta_epochs, base_inner_steps=inner_steps)


if __name__ == "__main__":
    main()
