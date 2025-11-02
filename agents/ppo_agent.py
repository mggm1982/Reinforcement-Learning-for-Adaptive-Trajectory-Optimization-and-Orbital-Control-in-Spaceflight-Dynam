from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from envs.orbit_env import RK4OrbitEnv

def make_env():
    def _init():
        return RK4OrbitEnv()
    return _init

def get_vectorized_env(n_envs=8):
    """Create a vectorized environment (does NOT auto-start)."""
    envs = [make_env() for _ in range(n_envs)]
    return SubprocVecEnv(envs)

def build_agent(n_envs=8, **kwargs):
    """Build PPO agent safely inside main."""
    env = get_vectorized_env(n_envs)
    model = PPO("MlpPolicy", env, verbose=1,
                learning_rate=kwargs.get("learning_rate", 3e-4),
                gamma=kwargs.get("gamma", 0.99),
                n_steps=2048,
                batch_size=kwargs.get("batch_size", 64),
                ent_coef=kwargs.get("ent_coef", 0.01),
                n_epochs=10)
    return model, env
