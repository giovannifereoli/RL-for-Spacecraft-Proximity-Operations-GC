from stable_baselines3.common.callbacks import BaseCallback

class CallBack(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, env, verbose=0):
        super(CallBack, self).__init__(verbose)
        self.env = env

    def _on_step(self):
        pass

    def _on_rollout_end(self):
        self.env.reset()
        pass
