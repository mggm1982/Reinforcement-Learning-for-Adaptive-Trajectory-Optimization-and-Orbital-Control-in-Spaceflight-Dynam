import copy
import torch
import yaml
from stable_baselines3 import PPO
from agents.ppo_agent import get_vectorized_env


def meta_train(meta_epochs: int = 5, inner_steps: int = 200_000):
    """
    Meta-training loop for Safe Meta-RL PPO.
    Performs MAML-style averaging of adapted PPO agents across tasks.
    """

    # -------------------------------
    # 1. Load and sanitize configuration
    # -------------------------------
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Force all numeric values into correct native Python types
    cfg["learning_rate"] = float(cfg.get("learning_rate", 3e-4))
    cfg["gamma"] = float(cfg.get("gamma", 0.99))
    cfg["batch_size"] = int(cfg.get("batch_size", 64))
    cfg["ent_coef"] = float(cfg.get("ent_coef", 0.01))
    cfg["n_envs"] = int(cfg.get("n_envs", 8))

    print(f"⚙️ Meta-RL Config: {cfg}")

    # -------------------------------
    # 2. Define meta-tasks
    # -------------------------------
    tasks = [
        {"name": "hohmann"},
        {"name": "drag_perturb"}
    ]

    # -------------------------------
    # 3. Initialize base (meta) model
    # -------------------------------
    env = get_vectorized_env(cfg["n_envs"])

    # Use a callable for LR to guarantee type-safety for SB3
    meta_model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=lambda _: cfg["learning_rate"],
        gamma=cfg["gamma"],
        batch_size=cfg["batch_size"],
        ent_coef=cfg["ent_coef"],
        n_steps=2048,
        n_epochs=10,
        device="auto"
    )

    # -------------------------------
    # 4. Meta-training loop
    # -------------------------------
    for epoch in range(meta_epochs):
        adapted_models = []

        for task in tasks:
            print(f"\n🧩 Adapting to task: {task['name']}")

            env_task = get_vectorized_env(cfg["n_envs"])
            model_task = PPO(
                "MlpPolicy",
                env_task,
                verbose=0,
                learning_rate=lambda _: cfg["learning_rate"],
                gamma=cfg["gamma"],
                batch_size=cfg["batch_size"],
                ent_coef=cfg["ent_coef"],
                n_steps=2048,
                n_epochs=10,
                device="auto"
            )

            # Copy meta-policy weights into task model
            model_task.policy.load_state_dict(copy.deepcopy(meta_model.policy.state_dict()))

            # Inner loop: train task-specific agent
            model_task.learn(total_timesteps=inner_steps)
            adapted_models.append(model_task)

        # -------------------------------
        # 5. Meta-update (parameter averaging)
        # -------------------------------
        with torch.no_grad():
            for p_meta, *p_tasks in zip(
                meta_model.policy.parameters(),
                *[m.policy.parameters() for m in adapted_models]
            ):
                p_meta.copy_(torch.mean(torch.stack([p.data for p in p_tasks]), dim=0))

        print(f"✅ Meta epoch {epoch + 1}/{meta_epochs} complete.")
        meta_model.save(f"meta_epoch_{epoch + 1}.zip")

    print("\n🏁 Meta-training complete.")
    return meta_model
