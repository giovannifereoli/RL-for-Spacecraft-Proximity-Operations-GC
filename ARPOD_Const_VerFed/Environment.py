# Import libraries
import random
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
        x0ivp=np.zeros(13),
        x0ivp_std=np.zeros(13),
        ang_corr=np.rad2deg(15),
        safety_radius=1,
        safety_vel=0.1,
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
        self.ang_corr = ang_corr
        self.safety_radius = safety_radius
        self.safety_vel = safety_vel
        self.rad_kso = 200
        self.rho_max = rho_max
        self.rhodot_max = rhodot_max
        self.infos = {"Episode success": "lost"}
        self.done = False
        self.Told = np.zeros(3)
        self.randomc = random.choice([1, 2, 3, 4])
        self.randomT = np.ones(3)
        if self.randomc != 4:
            self.randomT[self.randomc - 1] = 0.75

        # STATE AND ACTION SPACES
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1.25, high=+1.25, shape=(16,), dtype=np.float64
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
                1.2 * x0ivp[-2],
                0,
                -self.max_thrust,
                -200,
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
                0.8 * x0ivp[-2],  # OSS: empirically determined
                self.max_time,
                self.max_thrust,
                200,  # OSS: empirically determined
            ]
        ).flatten()

        # INITIAL CONDITIONS
        print("Initialization")
        # Part 1: get Initial Reward
        self.state0 = np.concatenate([x0ivp, np.array([0, 0])])  # Adding T and R initial states
        self.state0_std = np.concatenate([x0ivp_std, np.array([0, 0])])  # OSS: no std for T and R
        self.state = np.random.normal(
            self.state0, self.state0_std
        )  # OSS: not normalized as first step
        self.reward_old = self.get_reward(
            np.array([0, 0, 0]),
            self.state[6:-4],  # OSS: extracting relative state
            self.state[6:-4],
        )  # OSS: Reward t-1 without action

        # Part 2: get Initial State
        self.state0 = np.concatenate([x0ivp, np.array([0, self.reward_old])])
        self.state0_std = np.concatenate([x0ivp_std, np.array([0, 0])])
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
            std = 1e-12  # TODO: 1e-8 / mass
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

            # Chaser relative equations
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
                + Tx / m
                + np.random.uniform(0, std / (self.l_star / self.t_star**2)),
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
                + Ty / m
                + np.random.uniform(0, std / (self.l_star / self.t_star**2)),
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
                + Tz / m
                + np.random.uniform(0, std / (self.l_star / self.t_star**2)),
            ]
            dxdt[12] = -T_norm / (spec_impulse * g0)

            return dxdt

        # ACTUATION CONTROL
        # Thrust action with 50% failure in a random direction
        T = self.scaler_reverse_action(action) * self.randomT  # Actions t-1

        # EQUATIONS OF MOTION
        # Initialization
        x0 = self.scaler_reverse_observation(obs_scaled=self.state).flatten()

        # Integration
        sol = solve_ivp(
            fun=rel_crtbp,
            t_span=(0, self.dt),
            y0=x0[0:-3],  # x0 IVP != x0 MDP
            t_eval=[self.dt],
            method="LSODA",
            rtol=2.220446049250313e-14,
            atol=2.220446049250313e-14,
            args=(T, self.mu, self.spec_impulse, self.g0),  # OSS: it shall be a tuple
        )
        self.time += self.dt

        # Definition of complete MDP state from IVP state
        self.state = np.transpose(sol.y).flatten()
        self.state = np.append(self.state, self.max_time - self.time)
        self.state = np.append(self.state, np.linalg.norm(T))
        self.state = np.append(
            self.state, self.reward_old
        )  # OSS: reward of previous time-step

        # REWARD
        reward = self.get_reward(T, self.state[6:-4], x0[6:-4])  # Reward t due to actions t-1
        self.reward_old = reward  # OSS: update reward old, it has already been inserted in state

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
        # Random thrust failure
        self.randomc = random.choice([1, 2, 3, 4])
        self.randomT = np.ones(3)
        if self.randomc != 4:
            self.randomT[self.randomc - 1] = 0.75

        # Miscellaneous
        self.infos = {"Episode success": "lost"}
        self.done = False
        self.time = 0

        # Set initial conditions (OSS: already normalized)
        print("New initial condition")
        self.state = np.random.normal(self.state0, self.state0_std).flatten()
        self.reward_old = self.get_reward(np.array([0, 0, 0]), self.state[6:-4], self.state[6:-4])
        self.state = self.scaler_apply_observation(self.state)

        return self.state

    def get_reward(self, T, xrel_new, xrel_old):  # OSS: self_state =ish xrel_new
        # Useful data
        x_norm = np.linalg.norm(
            np.array(
                [
                    xrel_new[0:3] * self.l_star / self.rho_max,
                    xrel_new[3:6] * self.l_star / (self.t_star * self.rhodot_max),
                ]
            )
        ) / np.linalg.norm(np.array([1, 1, 1, 1, 1, 1]))
        x_norm_new = np.linalg.norm(
            np.array(
                [
                    xrel_new[0:3] * self.l_star / self.rho_max,
                    xrel_new[3:6] * self.l_star / (self.t_star * self.rhodot_max),
                ]
            )
        ) / np.linalg.norm(np.array([1, 1, 1, 1, 1, 1]))
        x_norm_old = np.linalg.norm(
            np.array(
                [
                    xrel_old[0:3] * self.l_star / self.rho_max,
                    xrel_old[3:6] * self.l_star / (self.t_star * self.rhodot_max),
                ]
            )
        ) / np.linalg.norm(np.array([1, 1, 1, 1, 1, 1]))
        rho = np.linalg.norm(xrel_new[0:3]) * self.l_star
        rhodot = np.linalg.norm(xrel_new[3:6]) * self.l_star / self.t_star
        T_norm = np.linalg.norm(
            T
        )  # OSS: not-scaled, already fine with self.max_thrust!
        print("Position %.4f m, velocity %.4f m/s" % (rho, rhodot))

        # TODO: is collided fa veramente qualcosa? o devo cambiare to angle version?

        # Dense reward RVD
        # reward = 1.5 * (np.linalg.norm(x_norm_old) - np.linalg.norm(x_norm_new))
        reward = (1 / 50) * np.log(x_norm) ** 2
        if rho >= self.rho_max:
            self.done = True
            reward += - 30  # OSS: Episodic RVD is useless, without you achieve also station-keeping.
        self.infos = {"Episode success": "approaching"}  # TODO: metti i constraint a -100 magari
        if rho <= self.safety_radius and rhodot <= self.safety_vel:
            self.infos = {"Episode success": "docked"}
            print('Docked.')
            reward += 100
            self.done = True

        # Dense reward constraints
        reward += self.is_outside3(rho, xrel_new)
        reward += self.T_const(T)

        # Dense reward thrust optimization
        reward += - (1 / 100) * np.exp(T_norm / self.max_thrust) ** 2

        # Scaling reward
        reward = reward / 50

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
        return obs_scaled

    # Remove scalers
    def scaler_reverse_observation(self, obs_scaled):
        obs = ((1 + obs_scaled) * (self.max - self.min)) / 2 + self.min
        return obs

    def is_outside(self, rho, xrel_new):
        # Initialization (matrix for +y-axis approach corridor)
        pos_vec = xrel_new[0:3] * self.l_star
        cone_vec = np.array([0, 1, 0])
        len_cut = np.sqrt((self.safety_radius**2) / np.square(np.tan(self.ang_corr)))
        const = -np.dot(pos_vec, cone_vec) + rho * np.cos(
            self.ang_corr
        )  # OSS: inside [rho (cos-1), rho(cos)]= rho[-0.03, 0.96]
        const2 = -np.dot(pos_vec + np.array([0, len_cut, 0]), cone_vec) + rho * np.cos(
            self.ang_corr  # TODO: è giusto?
        )

        # Computation reward (OSS: if B*x>0 constraint violated)
        reward_cons = - (1 / 10) * np.exp(0.5 * const / self.rho_max) ** 2

        # Computation collision with Truncated Cone (OSS: cone for reward != cone for collision signal)
        if const2 > 0:  # OSS: just a signal
            self.infos = {"Episode success": "collided"}
            print("Collision.")
            reward_cons += - 30
            self.done = True

        return reward_cons

    def is_outside3(self, rho, xrel_new):
        # Initialization (matrix for +y-axis approach corridor)
        pos_vec = xrel_new[0:3] * self.l_star
        cone_vec = np.array([0, 1, 0])
        ang = np.arccos(np.dot(pos_vec, cone_vec) / rho)
        len_cut = np.sqrt((self.safety_radius**2) / np.square(np.tan(self.ang_corr)))
        const2 = - np.dot(pos_vec + np.array([0, len_cut, 0]), cone_vec) + rho * np.cos(
            self.ang_corr
        )

        # Computation reward w.r.t. angle
        reward_cons = - (1 / 10) * np.exp(ang / (2 * np.pi)) ** 2

        # Computation collision
        if const2 > 0:  # and rho > 1.5:  # OSS: if B*x>0 constraint violated
            self.infos = {"Episode success": "collided"}
            print("Collision.")
            reward_cons += - 30
            self.done = True

        return reward_cons

    def T_const(self, Tnew):
        # Angular velocity
        Tnew_dir = Tnew / (np.linalg.norm(Tnew) + 1e-36)
        Told_dir = self.Told / (np.linalg.norm(self.Told) + 1e-36)
        dTdt_ver = (Tnew_dir - Told_dir) / (self.dt * self.l_star)  # Finite differences
        Tb_ver = np.array([1, 0, 0])
        wy = dTdt_ver[2] / Tb_ver[0]
        wz = - dTdt_ver[1] / Tb_ver[0]
        w_ang = np.linalg.norm(np.array([0, wy, wz]))  # TODO: perchè questo è diverso da simulazione post-training?
        if w_ang > np.deg2rad(10):
            print('High angular velocity.')

        reward_w = - (1 / 10) * np.exp(w_ang / (2 * np.pi)) ** 2  # TODO: ha senso questo?

        # Update Told
        self.Told = Tnew

        return reward_w

    def render(self, mode="human"):
        pass
