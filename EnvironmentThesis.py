# Import libraries
import gym
from gym import spaces
import numpy as np
from scipy.integrate import solve_ivp


class CustomEnv(gym.Env):
    # Initialize class
    def __init__(self):
        super(CustomEnv, self).__init__()
        # DATA
        self.mu = 0.012150583925359
        self.m_star = 6.0458 * 1e24  # Kilograms
        self.l_star = 3.844 * 1e8  # Meters
        self.t_star = 375200  # Seconds
        self.time = 0
        self.max_time = 20 / self.t_star
        self.dt = 0.25 / self.t_star
        self.max_thrust = 29620 / (self.m_star * self.l_star / self.t_star**2)
        self.spec_impulse = 310 / self.t_star
        self.g0 = 9.81 / (self.l_star / self.t_star**2)
        self.rho_max = 60
        self.rhodot_max = 6
        self.infos = {"Episode success": "lost"}
        self.done = False

        # STATE AND ACTION SPACES
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1.5, high=+1.5, shape=(13,), dtype=np.float64
        )

        # SCALERS
        # Initialization
        self.min = np.concatenate(
            (
                np.array(
                    [
                        392882528.83007175 / self.l_star,
                        - 13436.060445606101 / self.l_star,
                        - 69999240.0 / self.l_star,
                        - 0.045812455061369245 / (self.l_star / self.t_star),
                        - 105.88740080490405 / (self.l_star / self.t_star),
                        0.00259690377380597 / (self.l_star / self.t_star),
                        - self.rho_max / self.l_star,
                        - self.rho_max / self.l_star,
                        - self.rho_max / self.l_star,
                        - self.rhodot_max / (self.l_star / self.t_star),
                        - self.rhodot_max / (self.l_star / self.t_star),
                        - self.rhodot_max / (self.l_star / self.t_star),
                    ]
                ),
                np.array([20850 / self.m_star]),
            )
        ).flatten()
        self.max = np.concatenate(
            (
                np.array(  # 200 meters
                    [
                        392882531.736 / self.l_star,
                        - 200.136451174 / self.l_star,
                        - 69999228.94132772 / self.l_star,
                        - 0.0006823980728837953 / (self.l_star / self.t_star),
                        - 105.88737464056967 / (self.l_star / self.t_star),
                        0.17434185693104906 / (self.l_star / self.t_star),
                        self.rho_max / self.l_star,
                        self.rho_max / self.l_star,
                        self.rho_max / self.l_star,
                        self.rhodot_max / (self.l_star / self.t_star),  # TODO: forse è questo che ha stabilizzato
                        self.rhodot_max / (self.l_star / self.t_star),
                        self.rhodot_max / (self.l_star / self.t_star),
                    ]
                ),
                np.array([21100 / self.m_star]),  # TODO: prova ad aggiungere tempo qua
            )
        ).flatten()
        '''
        self.min = np.zeros(13).flatten()
        self.max = np.absolute(
            np.concatenate(
                (
                    np.array(  # 200 meters target
                        [
                            np.linalg.norm(x0t_state_2small[0:3]) / self.l_star,
                            np.linalg.norm(x0t_state_2small[0:3]) / self.l_star,
                            np.linalg.norm(x0t_state_2small[0:3]) / self.l_star,
                            np.linalg.norm(x0t_state_2small[3:6])
                            / (self.l_star / self.t_star),
                            np.linalg.norm(x0t_state_2small[3:6])
                            / (self.l_star / self.t_star),
                            np.linalg.norm(x0t_state_2small[3:6])
                            / (self.l_star / self.t_star),
                            self.rho_max / self.l_star,
                            self.rho_max / self.l_star,
                            self.rho_max / self.l_star,
                            self.rhodot_max / (self.l_star / self.t_star),
                            self.rhodot_max / (self.l_star / self.t_star),
                            self.rhodot_max / (self.l_star / self.t_star),
                        ]
                    ),
                    np.array([np.linalg.norm(x0r_mass) / self.m_star]),
                )
            )
        ).flatten()'''

        # IINITIAL CONDITIONS
        x0t_state_2small = np.array(
            [
                1.02206694e00,
                -1.33935003e-07,
                -1.82100000e-01,
                -1.71343849e-07,
                -1.03353155e-01,  # TODO: check tutti questi numerini
                6.52058535e-07,
            ]
        )  # 9:2 NRO - 50m after apolune, already corrected, rt = 399069639.7170633, vt = 105.88740083894766
        x0r_state_2small = np.array(
            [
                1.11022302e-13,
                1.33935003e-07,
                -4.22495372e-13,
                1.71343849e-07,
                -3.75061093e-13,
                -6.52058535e-07,
            ]
        )
        x0r_mass = np.array([21000 / self.m_star])
        self.state0 = np.concatenate((x0t_state_2small, x0r_state_2small, x0r_mass))
        self.state0_std = np.absolute(
            np.concatenate((np.zeros(6), 0.1 * x0r_state_2small, 0.005 * x0r_mass))
        )
        self.state = self.scaler_apply(obs=self.state0)
        self.state0_stoch = self.scaler_apply(
            np.random.normal(self.state0, self.state0_std)
        )

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
            dxdt[12] = - T_norm / (spec_impulse * g0)  # TODO: sistema questo

            return dxdt

        # ACTUATION CONTROL
        # Thrust action
        T = self.max_thrust * action / np.linalg.norm(np.array([1, 1, 1]))

        # EQUATIONS OF MOTION
        # Initialization
        x0 = self.scaler_reverse(obs_scaled=self.state).flatten()

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

        # Dense reward
        reward = (1 / 50) * np.log(x_norm) ** 2
        self.infos = {"Episode success": "approaching"}
        print("Position %.4f m, velocity %.4f m/s" % (rho, rhodot))

        # Episodic reward
        if self.time > 0.98 * self.max_time and rho <= 1 and rhodot <= 0.2:  # OSS: molto meglio farlo andare a ToF finale costante, sia per RVD che per convergenza.
            self.infos = {"Episode success": "docked"}
            print("Successful docking.")
            reward += 10
            self.done = True

        # Time constraint
        if self.time >= self.max_time:
            self.done = True

        # Return scaled state
        self.state = self.scaler_apply(obs=self.state)

        return self.state, reward, self.done, self.infos

    # Reset between episodes
    def reset(self):
        # Set initial conditions
        print("New initial condition")
        self.state0_stoch = self.scaler_apply(
            np.random.normal(self.state0, self.state0_std).flatten()
        )
        self.state = self.state0_stoch

        # Miscellaneous
        self.infos = {"Episode success": "lost"}
        self.done = False
        self.time = 0

        return self.state  # reward, done, info can't be included

    # Apply scalers
    def scaler_apply(self, obs):
        obs_scaled = (-1 + 2 * (obs - self.min) / (self.max - self.min))
        # obs_scaled = (obs - self.min) / (self.max - self.min)
        return obs_scaled

    # Remove scalers
    def scaler_reverse(self, obs_scaled):
        obs = ((1 + obs_scaled) * (self.max - self.min)) / 2 + self.min
        # obs = (obs_scaled * (self.max - self.min)) + self.min
        return obs

    def render(self, mode="human"):
        pass
