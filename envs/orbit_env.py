import torch
import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Keep heavy astro imports out of module-level scope (do them locally in cowell_step)
# This avoids huge memory/cpu pressure when multiprocessing spawns workers.

class CowellOrbitEnv(gym.Env):
    """
    Reinforced/robust version of your environment:
     - Local imports for poliastro/astropy to avoid spawn-time heavy loads.
     - Proper unit handling: pass plain floats (km, km/s, km^3/s^2) to cowell.
     - thrust_perturbation returns a 3-vector acceleration (km/s^2).
     - RK4 fallback integrator (SI units, uses self.mu in m^3/s^2).
    """

    def __init__(self, mu=3.986e14, dt=1.0, max_steps=4000):
        super().__init__()
        # mu in SI (m^3 / s^2)
        self.mu = float(mu)
        self.dt = float(dt)
        self.max_steps = int(max_steps)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        # state: [rx, ry, rz, vx, vy, vz, mass] (we store in SI (m, m/s) internally)
        self.observation_space = spaces.Box(low=-1e9, high=1e9, shape=(7,), dtype=np.float64)

        self.reset()

    # -------------------------------
    # INITIALIZATION
    # -------------------------------
    def _get_initial_state(self):
        # Keep original SI initial values (meters, m/s)
        r0 = torch.tensor([6.7e6, 0.0, 0.0], dtype=torch.float64)
        v0 = torch.tensor([0.0, 7700.0, 0.0], dtype=torch.float64)
        m0 = torch.tensor([500.0], dtype=torch.float64)
        return torch.cat([r0, v0, m0])

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.steps = 0
        self.state = self._get_initial_state()

        # target orbit radius (m) and velocity (m/s)
        self.target_radius = 7.2e6
        self.target_velocity = np.sqrt(self.mu / self.target_radius)
        self.target = torch.tensor(
            [self.target_radius, 0, 0, 0, self.target_velocity, 0, 500.0],
            dtype=torch.float64
        )

        return self._get_observation(), {}

    # -------------------------------
    # RK4 FALLBACK (SI: m, m/s)
    # -------------------------------
    def _rk4_fallback(self, state, action):
        """
        Simple RK4 integrator for one step using Newtonian gravity + thrust.
        State is torch tensor in SI (m, m/s, kg).
        action is torch tensor (3,) unitless control vector - scaled into acceleration.
        """
        dt = self.dt
        mu = self.mu

        r = state[:3].numpy()
        v = state[3:6].numpy()
        m = float(state[6].item())

        # convert action -> acceleration in m/s^2 (small magnitude)
        # (tunable scaling factor kept small)
        acc_action = (1e-6 * action.numpy() / max(m, 1.0))  # unit: (m/s^2) nominal

        def deriv(r_vec, v_vec, a_thrust):
            norm_r = np.linalg.norm(r_vec)
            if norm_r == 0:
                grav = np.zeros(3)
            else:
                grav = -mu * r_vec / (norm_r**3)
            return np.array([v_vec, grav + a_thrust], dtype=object)

        # RK4 steps (vector form)
        # k1
        a1 = acc_action
        k1_r = v
        k1_v = (-mu * r / (np.linalg.norm(r) ** 3)) + a1

        # k2
        r2 = r + 0.5 * dt * k1_r
        v2 = v + 0.5 * dt * k1_v
        k2_r = v2
        k2_v = (-mu * r2 / (np.linalg.norm(r2) ** 3)) + a1

        # k3
        r3 = r + 0.5 * dt * k2_r
        v3 = v + 0.5 * dt * k2_v
        k3_r = v3
        k3_v = (-mu * r3 / (np.linalg.norm(r3) ** 3)) + a1

        # k4
        r4 = r + dt * k3_r
        v4 = v + dt * k3_v
        k4_r = v4
        k4_v = (-mu * r4 / (np.linalg.norm(r4) ** 3)) + a1

        new_r = r + (dt / 6.0) * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)
        new_v = v + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

        new_m = m - np.linalg.norm(action.numpy()) * 0.05

        new_state = torch.tensor(
            np.concatenate([new_r, new_v, np.array([new_m])]), dtype=torch.float64
        )
        return new_state

    # -------------------------------
    # COWELL PROPAGATION STEP (robust)
    # -------------------------------
    def cowell_step(self, state, action):
        """
        Integrate one step using poliastro.cowell if available and the units are correct.
        We pass plain floats (km and km/s) to poliastro to avoid astropy units conversion problems.
        If poliastro fails for any reason, we fall back to the RK4 integrator above (SI units).
        """

        # extract SI state (m, m/s, kg)
        r_si = state[:3].numpy()        # meters
        v_si = state[3:6].numpy()       # m/s
        m = float(state[6].item())      # kg

        # Convert SI -> poliastro-friendly units: km and km/s
        r_km = r_si / 1000.0            # km
        v_kms = v_si / 1000.0           # km/s
        # poliastro expects gravitational parameter in km^3 / s^2 when units are km
        try:
            # local import to avoid heavy module import at file load time (helps multiprocessing spawn)
            from poliastro.twobody.propagation import cowell as poli_cowell
            from poliastro.bodies import Earth
            import astropy.units as u  # noqa: F401 (we only use for safe conversion below if needed)
        except Exception:
            # If import fails, do the simple rk4 fallback (SI).
            return self._rk4_fallback(state, action)

        # use Earth.k but extract numeric km^3/s^2
        try:
            k_q = Earth.k  # quantity (km^3 / s^2)
            k_val = float(k_q.to(u.km**3 / u.s**2).value)
        except Exception:
            # If conversion fails, fallback to numeric mu (converted from SI to km^3/s^2)
            k_val = self.mu / 1000.0**3  # m^3/s^2 -> km^3/s^2

        # ensure action is numpy float array
        action_np = np.array(action.numpy(), dtype=np.float64).flatten()

        # thrust perturbation: poliastro's cowell expects an 'ad' function that returns a 3-vector:
        # ad(t, u, k) -> acceleration components (ax, ay, az) in same length units/time^2 (here km/s^2)
        def thrust_perturbation(t0, u_vec, k_):
            # u_vec: [x, y, z, vx, vy, vz] in (km, km/s)
            # We'll produce acceleration in km/s^2
            # scale action into small acceleration (this scaling can be tuned)
            # using mass in kg -> a = scale * action / mass (units km/s^2)
            scale = 1e-6  # unitless tuning constant
            # produce acceleration in km/s^2
            a_kms2 = (scale * action_np) / max(m, 1.0)  # km/s^2 (small)
            # ensure shape (3,)
            return np.asarray(a_kms2, dtype=np.float64)

        tof = self.dt  # seconds (float, no units)
        try:
            # Call poliastro cowell with plain numeric arrays (km, km/s, numeric k)
            r_new_km, v_new_kms = poli_cowell(k_val, r_km, v_kms, tof, ad=thrust_perturbation)
            # poli_cowell returns arrays (if inputs were plain numbers)
            # Convert back to SI (m, m/s)
            new_r_si = np.asarray(r_new_km, dtype=np.float64) * 1000.0
            new_v_si = np.asarray(v_new_kms, dtype=np.float64) * 1000.0
            new_m = m - np.linalg.norm(action_np) * 0.05
            new_state = torch.tensor(
                np.concatenate([new_r_si, new_v_si, np.array([new_m])]), dtype=torch.float64
            )
            return new_state

        except Exception:
            # If poliastro fails for any reason (units or other), fall back to RK4 (SI)
            try:
                return self._rk4_fallback(state, action)
            except Exception:
                # as last resort, raise so user sees underlying issue
                raise

    # -------------------------------
    # STEP + REWARD
    # -------------------------------
    def _get_observation(self):
        # return numpy array (float64)
        return self.state.numpy()

    def step(self, action):
        """
        One environment step with a clean differential reward:
        - distance progress (primary)
        - energy progress (secondary)
        - small prograde alignment bonus
        - fuel/action penalty
        - heavy penalty + terminate if below LEO limit
        - terminal bonus on successful circularization
        """

        # --- bookkeeping ---
        self.steps += 1

        # ensure tensors and detach previous state snapshot
        action = torch.tensor(action, dtype=torch.float64).flatten()
        prev_state = self.state.clone().detach()
        prev_r = torch.norm(prev_state[:3])
        prev_v = torch.norm(prev_state[3:6])
        prev_m = prev_state[6].clone().detach()

        # numeric targets as torch tensors (for consistent ops)
        target_radius_t = torch.tensor(self.target_radius, dtype=torch.float64)
        mu_t = torch.tensor(self.mu, dtype=torch.float64)

        # previous distance & energy errors
        prev_dist_err = torch.abs(prev_r - target_radius_t)
        prev_energy = 0.5 * prev_v ** 2 - mu_t / prev_r
        target_energy = -mu_t / (2.0 * target_radius_t)
        prev_energy_err = torch.abs(prev_energy - target_energy) / (torch.abs(target_energy) + 1e-12)

        # propagate using cowell_step (returns torch tensor in SI: m, m/s, kg)
        next_state = self.cowell_step(prev_state, action)

        # current orbital scalars
        r = torch.norm(next_state[:3])
        v = torch.norm(next_state[3:6])
        mass = next_state[6].clone().detach()

        dist_to_target = torch.abs(r - target_radius_t)
        energy = 0.5 * v ** 2 - mu_t / r
        energy_err = torch.abs(energy - target_energy) / (torch.abs(target_energy) + 1e-12)

        # --- differential progress terms (positive = good) ---
        dist_reduction = (prev_dist_err - dist_to_target)            # meters; >0 if we got closer
        energy_reduction = (prev_energy_err - energy_err)           # dimensionless ratio improvement (>0 good)

        # --- alignment / prograde bonus ---
        # Compute alignment between thrust direction (action) and velocity vector at the *new* state.
        # We expect prograde burns (action aligned with velocity) to be helpful.
        vel_vec = next_state[3:6]
        act_norm = torch.norm(action) + 1e-12
        vel_norm = torch.norm(vel_vec) + 1e-12
        cos_align = torch.dot(action, vel_vec) / (act_norm * vel_norm)
        # keep only positive (prograde) alignment
        prograde_align = torch.clamp(cos_align, min=0.0)

        # --- fuel / action penalty (we prefer smaller impulses) ---
        # estimate fuel used (keep same scaling as cowell_step)
        fuel_used = act_norm * 0.05  # same scale as cowell_step mass subtraction

        # --- weights (simple, tuned to magnitudes) ---
        w_dist = 1.0            # per meter-term will be scaled below
        w_energy = 5.0          # energy ratio importance
        w_align = 50.0          # alignment is important but applied to small values
        w_fuel = 20.0           # heavy penalty for fuel usage
        w_action = 0.0          # optional direct action penalty (kept 0; fuel covers it)
        w_reach = 150.0         # terminal success bonus
        # scale for distance (meters -> normalized)
        dist_scale = 1e4        # 10 km scale (so 1 unit ~ 10 km progress)

        # --- compute reward (differential + simple) ---
        reward = torch.tensor(0.0, dtype=torch.float64)

        # progress terms: normalized to reasonable ranges
        reward = reward + w_dist * (dist_reduction / dist_scale)
        reward = reward + w_energy * energy_reduction
        # prograde alignment gives an extra small bonus *only if making progress*
        prog_mask = (dist_reduction > 0).float()
        reward = reward + w_align * prograde_align * prog_mask * (dist_reduction / dist_scale)
        # fuel penalty (negative)
        reward = reward - w_fuel * fuel_used
        # small action penalty (if desired)
        reward = reward - w_action * act_norm

        # --- safety / termination rules ---
        # Extremely large negative reward and immediate termination if falling below safe LEO radius.
        low_leo_limit = 6.6e6  # meters (adjust to your chosen LEO floor)
        terminated = False
        truncated = False

        if r < low_leo_limit:
            # catastrophic penalty
            reward = torch.tensor(-1000.0, dtype=torch.float64)
            terminated = True

        # success condition (stable circularized orbit near target)
        dist_tol = 2e5        # 200 km tolerance (adjustable)
        vel_tol = 100.0       # m/s tolerance on velocity
        vel_target = torch.sqrt(mu_t / target_radius_t)
        vel_diff = torch.abs(v - vel_target)

        if (dist_to_target < dist_tol) and (vel_diff < vel_tol):
            reward = reward + w_reach
            terminated = True

        # fuel-out termination
        if mass < 100.0:
            terminated = True
            reward = reward - 500.0  # strong penalty for running out of fuel mid-transfer

        # time truncation
        if self.steps >= self.max_steps:
            truncated = True

        # finalize state, info, return types
        self.state = next_state.clone().detach()
        obs = self._get_observation()

        info = {
            "distance_to_target": float(dist_to_target.item()),
            "velocity_diff": float(vel_diff.item()),
            "mass": float(mass.item()),
            "dist_reduction": float(dist_reduction.item()),
            "energy_reduction": float(energy_reduction.item()),
            "prograde_align": float(prograde_align.item()),
            "fuel_used": float(fuel_used.item()),
        }

        # clamp reward to avoid extreme floating range issues (optional)
        reward = torch.clamp(reward, -1e4, 1e4)

        return obs, float(reward.item()), bool(terminated), bool(truncated), info
