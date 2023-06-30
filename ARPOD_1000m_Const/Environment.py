# Import libraries
import gym
from gym import spaces
import numpy as np
from scipy.integrate import solve_ivp


class ArpodCrtbp(gym.Env):
    # Initialize class
    def __init__(
        self,
        max_time=1,
        dt=1,
        rho_max=1,
        rhodot_max=1,
        x0=np.zeros(13),
        x0_std=np.zeros(13),
    ):
        super(ArpodCrtbp, self).__init__()
        # DATA
        self.mu = 0.012150583925359
        self.m_star = 6.0458 * 1e24  # Kilograms
        self.l_star = 3.844 * 1e8  # Meters
        self.t_star = 375200  # Seconds
        self.time = 0
        self.max_time = max_time / self.t_star
        self.dt = dt / self.t_star
        self.max_thrust = 29620 / (self.m_star * self.l_star / self.t_star**2)
        self.spec_impulse = 310 / self.t_star
        self.g0 = 9.81 / (self.l_star / self.t_star**2)
        self.ang_corr = np.deg2rad(10)
        self.rad_kso = 200
        self.rho_max = rho_max
        self.rhodot_max = rhodot_max
        self.infos = {"Episode success": "lost"}
        self.done = False

        # STATE AND ACTION SPACES
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1.25, high=+1.25, shape=(13,), dtype=np.float64
        )

        # SCALERS
        # Initialization (OSS: max-min target state taken from 9:2 NRO full orbit)
        self.min = np.array(
            [
                379548434.40513575 / self.l_star,
                -16223383.008425826 / self.l_star,
                -70002940.10058032 / self.l_star,
                -81.99561388926969 / (self.l_star / self.t_star),
                -105.88740121359594 / (self.l_star / self.t_star),
                -881.9954974936014 / (self.l_star / self.t_star),
                -self.rho_max / self.l_star,
                -self.rho_max / self.l_star,
                -self.rho_max / self.l_star,
                -self.rhodot_max / (self.l_star / self.t_star),
                -self.rhodot_max / (self.l_star / self.t_star),
                -self.rhodot_max / (self.l_star / self.t_star),
                1.2 * x0[-1],
            ]
        ).flatten()
        self.max = np.array(
            [
                392882530.7281463 / self.l_star,
                16218212.912172267 / self.l_star,
                3248770.078052207 / self.l_star,
                82.13051133777446 / (self.l_star / self.t_star),
                1707.5720010497114 / (self.l_star / self.t_star),
                881.8822374702228 / (self.l_star / self.t_star),
                self.rho_max / self.l_star,
                self.rho_max / self.l_star,
                self.rho_max / self.l_star,
                self.rhodot_max / (self.l_star / self.t_star),
                self.rhodot_max / (self.l_star / self.t_star),
                self.rhodot_max / (self.l_star / self.t_star),
                0.8 * x0[-1],
            ]
        ).flatten()

        # INITIAL CONDITIONS
        self.state0 = x0
        self.state0_std = x0_std
        self.state = self.scaler_apply_observation(
            np.random.normal(self.state0, self.state0_std)
        )
        # OSS: state is always normalized in the flow BESIDE during integration!

    # MDP step
    def step(self, action):
        # RELATIVE CRT3BP
        def rel_crtbp(
            t,
            x,
            T,
            mu=0.012150583925359,
            spec_impulse=310 / self.t_star,
            g0=9.81 / (self.l_star / self.t_star**2),
        ):
            """
                        Circular Restricted Three-Body Problem Dynamics
            :
                        :param t: time
                        :param x: State, vector 13x1
                        :param T: Thrust action
                        :param mu: Gravitational constant, scalar
                        :param spec_impulse: Specific impulse
                        :param g0: Constant
                        :return: State Derivative, vector 6x1
            """

            # Initialize ODE
            dxdt = np.zeros((13,))
            # Initialize Target State
            xt = x[0]
            yt = x[1]
            zt = x[2]
            xtdot = x[3]
            ytdot = x[4]
            ztdot = x[5]
            # Initialize Relative State
            xr = x[6]
            yr = x[7]
            zr = x[8]
            xrdot = x[9]
            yrdot = x[10]
            zrdot = x[11]
            # Initial Mass Target
            m = x[12]
            # Initialize Thrust action
            Tx = T[0]
            Ty = T[1]
            Tz = T[2]
            T_norm = np.linalg.norm(T)

            # Relative CRTBP Dynamics
            r1t = [xt + mu, yt, zt]
            r2t = [xt + mu - 1, yt, zt]
            r1t_norm = np.sqrt((xt + mu) ** 2 + yt**2 + zt**2)
            r2t_norm = np.sqrt((xt + mu - 1) ** 2 + yt**2 + zt**2) + t * 0
            rho = [xr, yr, zr]

            # Target Equations
            dxdt[0:3] = [xtdot, ytdot, ztdot]
            dxdt[3:6] = [
                2 * ytdot
                + xt
                - (1 - mu) * (xt + mu) / r1t_norm**3
                - mu * (xt + mu - 1) / r2t_norm**3,
                -2 * xtdot
                + yt
                - (1 - mu) * yt / r1t_norm**3
                - mu * yt / r2t_norm**3,
                -(1 - mu) * zt / r1t_norm**3 - mu * zt / r2t_norm**3,
            ]

            # Chaser equations
            dxdt[6:9] = [xrdot, yrdot, zrdot]
            dxdt[9:12] = [
                2 * yrdot
                + xr
                + (1 - mu)
                * (
                    (xt + mu) / r1t_norm**3
                    - (xt + xr + mu) / np.linalg.norm(np.add(r1t, rho)) ** 3
                )
                + mu
                * (
                    (xt + mu - 1) / r2t_norm**3
                    - (xt + xr + mu - 1) / np.linalg.norm(np.add(r2t, rho)) ** 3
                )
                + Tx / m,
                -2 * xrdot
                + yr
                + (1 - mu)
                * (
                    yt / r1t_norm**3
                    - (yt + yr) / np.linalg.norm(np.add(r1t, rho)) ** 3
                )
                + mu
                * (
                    yt / r2t_norm**3
                    - (yt + yr) / np.linalg.norm(np.add(r2t, rho)) ** 3
                )
                + Ty / m,
                (1 - mu)
                * (
                    zt / r1t_norm**3
                    - (zt + zr) / np.linalg.norm(np.add(r1t, rho)) ** 3
                )
                + mu
                * (
                    zt / r2t_norm**3
                    - (zt + zr) / np.linalg.norm(np.add(r2t, rho)) ** 3
                )
                + Tz / m,
            ]
            dxdt[12] = -T_norm / (spec_impulse * g0)

            return dxdt

        # ACTUATION CONTROL
        # Thrust action
        T = self.scaler_reverse_action(action)

        # EQUATIONS OF MOTION
        # Initialization
        x0 = self.scaler_reverse_observation(obs_scaled=self.state).flatten()

        # Integration
        sol = solve_ivp(
            fun=rel_crtbp,
            t_span=(0, self.dt),
            y0=x0,
            t_eval=[self.dt],
            method="LSODA",
            rtol=2.220446049250313e-14,
            atol=2.220446049250313e-14,
            args=(T, self.mu, self.spec_impulse, self.g0),  # OSS: it shall be a tuple
        )
        self.state = np.transpose(sol.y).flatten()
        self.time += self.dt

        # REWARD
        reward = self.get_reward(T)

        # Time constraint
        if self.time >= self.max_time:
            self.done = True

        # Return scaled state
        self.state = self.scaler_apply_observation(obs=self.state)

        return (
            self.state,
            reward,
            self.done,
            self.infos,
        )

    # Reset between episodes
    def reset(self):
        # Set initial conditions (OSS: already normalized)
        print("New initial condition")
        self.state = self.scaler_apply_observation(
            np.random.normal(self.state0, self.state0_std).flatten()
        )

        # Miscellaneous
        self.infos = {"Episode success": "lost"}
        self.done = False
        self.time = 0

        return self.state

    def get_reward(self, T):
        # Useful data
        x_norm = np.linalg.norm(
            np.array(
                [
                    self.state[6:9] * self.l_star / self.rho_max,
                    self.state[9:12] * self.l_star / (self.t_star * self.rhodot_max),
                ]
            )
        ) / np.linalg.norm(np.array([1, 1, 1, 1, 1, 1]))
        rho = np.linalg.norm(self.state[6:9]) * self.l_star
        rhodot = np.linalg.norm(self.state[9:12]) * self.l_star / self.t_star
        T_norm = np.linalg.norm(T)  # OSS: not-scaled, already fine with self.max_thrust!
        print("Position %.4f m, velocity %.4f m/s" % (rho, rhodot))

        # Dense reward RVD
        reward = (1 / 100) * np.log(x_norm) ** 2
        self.infos = {"Episode success": "approaching"}
        if rho <= 1 and rhodot <= 0.1:
            self.infos = {"Episode success": "docked"}

        # Dense reward constraints
        reward += self.is_outside()

        # Dense reward thrust optimization
        # reward += - 0.1 * T_norm / self.max_thrust

        return reward

    # Re-scale action from policy net
    def scaler_reverse_action(self, action):
        action_notscaled = (
            self.max_thrust * action / np.linalg.norm(np.array([1, 1, 1]))
        )
        return action_notscaled

    # Apply scalers
    def scaler_apply_observation(self, obs):
        obs_scaled = -1 + 2 * (obs - self.min) / (self.max - self.min)
        return obs_scaled  # TODO: check qua scalers

    # Remove scalers
    def scaler_reverse_observation(self, obs_scaled):
        obs = ((1 + obs_scaled) * (self.max - self.min)) / 2 + self.min
        return obs

    def is_outside(self):
        # Initialization (matrix for +y-axis approach corridor)
        B_const = np.array(
            [
                [- 1, - np.tan(self.ang_corr), 0],
                [1, - np.tan(self.ang_corr), 0],
                [0, - np.tan(self.ang_corr), 1],
                [0, - np.tan(self.ang_corr), - 1],
            ]
        )
        pos_vec = self.state[6:9] * self.l_star
        reward_cons = 0

        # Computation
        if np.any(np.dot(B_const, pos_vec) > 0):  # OSS: if B*x>0 constraint violated
            reward_cons = - 2 * np.exp(0.5 * np.max(np.dot(B_const, pos_vec)) / self.rho_max) ** 2
            self.infos = {"Episode success": "collided"}
            print("Collision.")

        return reward_cons

    def render(self, mode="human"):
        pass
