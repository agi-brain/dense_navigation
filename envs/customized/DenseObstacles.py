import numpy as np
import pybullet

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from envs.customized.BaseSingleAgentAviary import ActionType, ObservationType, BaseSingleAgentAviary


class DenseObstacles(BaseSingleAgentAviary):
    """Single agent RL problem: hover at position."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,  # TODO: if you want to set a fixed start pos, set as there, format as[[x, y, z]]
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=True,  # NOTE: There need to be set as False if you want to gain time
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.VEL,
                 use_random_start: bool = False,  # TODO
                 use_random_goal: bool = False
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        # TODO: Add Obstacle
        goal : list of (x, y, z), i think it x and y in workspace, and z can set as constant.
        """
        super().__init__(drone_model=drone_model,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act,
                         use_random_start=use_random_start,
                         use_random_goal=use_random_goal
                         )

    def _checkCollision(self):

        collision2obstacles = []
        for i in range(len(self.obstacleId)):
            close_info = pybullet.getClosestPoints(self.DRONE_IDS[0], self.obstacleId[i], distance=0.005,
                                                   physicsClientId=self.CLIENT)
            if close_info:
                collision2obstacles.append(True)  # make collision
            else:
                collision2obstacles.append(False)  # no collision

        return any(collision2obstacles)
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """

        state = self._getDroneStateVector(0)
        collisionFlag = self._checkCollision()
        current_distance = np.linalg.norm(np.array(self.goalPos) - state[0:3])
        if 1000 * (self.previous_distance_to_goal - current_distance) < 0.3:
            distance_reward = (max(0, (13 - current_distance ** 2)/13))/4
        else:
            distance_reward = max(0, (13 - current_distance ** 2)/13)
        self.previous_distance_to_goal = current_distance
        step_penalty = 0
        if collisionFlag:
            collision_penalty = -10
        else:
            collision_penalty = 0
        if np.linalg.norm(np.array(self.goalPos) - state[0:3]) < 0.08:
            signal_reward = 8500
        else:
            signal_reward = 0
        reward = signal_reward + distance_reward + step_penalty + collision_penalty
        return reward


    ################################################################################

    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        collisionFlag = self._checkCollision()
        if np.linalg.norm(np.array(self.goalPos) - state[0:3]) < 0.08:
            return True
        elif collisionFlag:
            return True
        elif state[2] > 0.49:
            return True
        elif not (-1.1 < state[0] < 1.2 and -2.1 < state[1] < 1.2):
            return True
        else:
            return False

    ################################################################################

    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        if self.step_counter > 9000:
            return True
        else:
            return False

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42}  #### Calculated by the Deep Thought supercomputer in 7.5M years
