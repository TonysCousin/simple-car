from distutils.log import debug
import math
import gym
from gym.spaces import Discrete, Box
import numpy as np
from ray.rllib.env.env_context import EnvContext

class Car(gym.Env):  #Based on OpenAI gym 0.26.1 API

    metadata = {"render_modes": None}
    #metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    OBS_SIZE                = 5
    MAX_SPEED               = 35.0      #vehicle's max achievable speed, m/s
    MAX_ACCEL               = 3.0       #vehicle's max achievable acceleration (fwd backward)
    ROAD_SPEED_LIMIT        = 29.1      #Roadway's legal speed limit, m/s
    SCENARIO_LENGTH         = 2000.0    #total length of the roadway, m
    SCENARIO_BUFFER_LENGTH  = 200.0     #length of buffer added to the end of continuing lanes, m


    def __init__(self,
                 config:        EnvContext,             #dict of config params
                 seed:          int             = None, #seed for PRNG
                 render_mode:   int             = None  #Ray rendering info, unused in this version
                ):
        """Initialize an object of this class.  Config options are:
            time_step_size: duration of a time step, s (default = 0.5)
            training:       (bool) True if this is a training run, else it is inference (affects initial conditions)
            init_ego_speed: initial speed of the agent vehicle (defaults to random in [0.1, 0.5])
            init_ego_x:     initial downtrack location of the agent vehicle from lane begin (defaults to random in [0, 0.2])
        """

        super().__init__()

        # Store the arguments
        self.prng = np.random.default_rng(seed = seed)
        self.render_mode = render_mode

        self.time_step_size = 0.5 #seconds
        try:
            ts = config["time_step_size"]
        except KeyError as e:
            ts = None
        if ts is not None  and  ts != ""  and  float(ts) > 0.0:
            self.time_step_size = float(ts)

        self._set_initial_conditions(config)

        # Indices into the observation vector (need to have all vehicles contiguous with ego being the first one)
        self.EGO_X              =  0 #agent's distance downtrack in that lane (center of bounding box), m
        self.EGO_SPEED          =  1 #agent's forward speed, m/s
        self.EGO_LANE_REM       =  2 #distance remaining in the agent's current lane, m
        self.EGO_ACCEL_CMD_CUR  =  3 #agent's most recent accel_cmd, m/s^2
        self.EGO_ACCEL_CMD_PREV =  4 #agent's accel cmd from prev time step, m/s^2

        # Obs vector only handles normalized values.  Distances are in [-1.1, 1.1], speeds are in [0, 1]
        # acceleration is in [-1, 1], curvatures are scaled logarithms in [-1, 1]
        lower_obs = np.zeros((Car.OBS_SIZE)) #most values are 0, so only the others are explicitly set below
        lower_obs[self.EGO_LANE_REM]        = -1.1
        lower_obs[self.EGO_ACCEL_CMD_CUR]   = -Car.MAX_ACCEL
        lower_obs[self.EGO_ACCEL_CMD_PREV]  = -Car.MAX_ACCEL

        upper_obs = np.ones(Car.OBS_SIZE)
        upper_obs[self.EGO_X]               = 1.1
        upper_obs[self.EGO_SPEED]           = Car.MAX_SPEED
        upper_obs[self.EGO_LANE_REM]        = 1.1
        upper_obs[self.EGO_ACCEL_CMD_CUR]   = Car.MAX_ACCEL
        upper_obs[self.EGO_ACCEL_CMD_PREV]  = Car.MAX_ACCEL

        self.observation_space = Box(low=lower_obs, high=upper_obs, dtype=np.float32)

        lower_act = np.array([-1.0])
        upper_act = np.array([ 1.0])
        self.action_space = Box(low=lower_act, high=upper_act, dtype=np.float32)

        self.obs = np.zeros(Car.OBS_SIZE) #will be returned from reset() and step()

        # Other persistent data
        self.stopped_count = 0 #num consecutive time steps in an episode where vehicle speed is zero
        self.steps_since_reset = 0
        self.steps_since_init = 0

        self.window = None
        self.clock = None


    def seed(self, seed=None):
        """A required method that is apparently not yet supported in Ray 2.0.0."""
        pass


    def reset(self,
              seed:         int             = None,
              options:      dict            = None
             ) -> list:
        """Reinitializes the environment to prepare for a new episode.  This must be called before
            making any calls to step().
        """

        # If we are in a training run, then choose widely randomized initial conditions
        ego_x = None
        ego_speed = None
        if self.training:
            ego_x = 0.0
            ego_speed = self.prng.random() * Car.MAX_SPEED

        # Else, we are doing inference, so limit the randomness of the initial conditions
        else:
            ego_x = self.prng.random() * 0.2*Car.SCENARIO_LENGTH  if self.init_ego_x is None  else  self.init_ego_x
            ego_speed = (self.prng.random() * 0.4 + 0.1)*Car.MAX_SPEED  if self.init_ego_speed is None  else self.init_ego_speed
        ego_rem = Car.SCENARIO_LENGTH + Car.SCENARIO_BUFFER_LENGTH - ego_x

        # Reinitialize the whole observation vector
        self.obs = np.zeros(Car.OBS_SIZE)
        self.obs[self.EGO_X]                = ego_x / Car.SCENARIO_LENGTH
        self.obs[self.EGO_SPEED]            = ego_speed / Car.MAX_SPEED
        self.obs[self.EGO_LANE_REM]         = ego_rem / Car.SCENARIO_LENGTH

        # Other persistent data
        self.stopped_count = 0
        self.steps_since_reset = 0

        return self.obs


    def step(self,
                action  : list      #list of floats; the only element is acceleration, scaled in [-1, 1] from the NN
            ):
        """Executes a single time step of the environment.  Determines how the input actions will alter the
            simulated world and returns the resulting observations to the agent.

            NOTE: obs vector is scaled for NN consumption, but all other calcs are in real world units & magnitudes.
        """

        # Clip the scaled actions if desired
        clip = min(self.action_clip_init + (1.0 - self.action_clip_init) * self.steps_since_init / self.action_clip_timesteps, 1.0)
        for a in action:
            a = min(max(a, -clip), clip)

        new_accel = action[0] * Car.MAX_ACCEL
        assert -Car.MAX_ACCEL <= new_accel <= Car.MAX_ACCEL, "Input accel cmd invalid: {:.2f}".format(action[0])
        self.steps_since_reset += 1
        self.steps_since_init += 1
        done = False
        crash = False
        return_info = {"reason": "Unknown"}

        # Move the vehicle downtrack (need to do some scaling here)
        prev_accel = self.obs[self.EGO_ACCEL_CMD_CUR] * Car.MAX_ACCEL
        prev_speed = self.obs[self.EGO_SPEED] * Car.MAX_SPEED
        prev_dist = self.obs[self.EGO_X] * Car.SCENARIO_LENGTH
        new_ego_speed = min(max(prev_speed + self.time_step_size*new_accel, 0.0), Car.MAX_SPEED)
        new_ego_x = prev_dist + self.time_step_size*new_ego_speed
        new_ego_rem = Car.SCENARIO_LENGTH + Car.SCENARIO_BUFFER_LENGTH - new_ego_x

        # If the ego vehicle has run off the end of the scenario, consider the episode successfully complete
        if new_ego_x >= Car.SCENARIO_LENGTH:
            new_ego_x = Car.SCENARIO_LENGTH #clip it so it doesn't violate obs bounds
            done = True
            return_info["reason"] = "Success; end of scenario"

        # Update the obs vector with the new vehicle state info
        self.obs[self.EGO_ACCEL_CMD_PREV] = prev_accel / Car.MAX_ACCEL
        self.obs[self.EGO_ACCEL_CMD_CUR] = new_accel / Car.MAX_ACCEL
        self.obs[self.EGO_LANE_REM] = new_ego_rem / Car.SCENARIO_LENGTH
        self.obs[self.EGO_SPEED] = new_ego_speed / Car.MAX_SPEED
        self.obs[self.EGO_X] = new_ego_x / Car.SCENARIO_LENGTH

        # If vehicle has been stopped for several time steps, then declare the episode done as a failure
        stopped_vehicle = False
        if new_ego_speed < 0.1:
            self.stopped_count += 1
            if self.stopped_count > 3:
                done = True
                stopped_vehicle = True
                return_info["reason"] = "Vehicle chose to stop moving"
        else:
            self.stopped_count = 0

        # Determine the reward resulting from this time step's action
        reward, expl = self._get_reward(done, crash, stopped_vehicle)
        return_info["reward_detail"] = expl

        return self.obs, reward, done, return_info


    def close(self):
        """Closes any resources that may have been used during the simulation."""
        pass #this method not needed for this version


    ##### internal methods #####


    def _set_initial_conditions(self,
                                config:     EnvContext
                               ):
        """Sets the initial conditions of the ego vehicle in member variables (speed, downtrack position)."""

        self.training = False
        try:
            tr = config["training"]
            if tr:
                self.training = True
        except KeyError as e:
            pass

        self.init_ego_speed = None
        try:
            es = config["init_ego_speed"]
            if 0 <= es <= Car.MAX_SPEED:
                self.init_ego_speed = es
        except KeyError as e:
            pass

        self.init_ego_x = None
        try:
            ex = config["init_ego_x"]
            if 0 <= ex < 1.0:
                self.init_ego_x = ex
        except KeyError as e:
            pass

        self.action_clip_init = 1.0
        try:
            aci = config["action_clip_init"]
            if 0 < aci <= 1.0:
                self.action_clip_init = aci
        except KeyError as e:
            pass

        self.action_clip_timesteps = 1
        try:
            act = config["action_clip_timesteps"]
            if act > 0:
                self.action_clip_timesteps = int(act)
        except KeyError as e:
            pass


    def _get_reward(self,
                    done    : bool,         #is this the final step in the episode?
                    crash   : bool,         #has the vehicle crashed?
                    stopped : bool          #has the vehicle come to a standstill?
                   ):
        """Returns the reward for the current time step (float).  The reward should be in [-2, 2] for any situation."""

        reward = 0.0
        explanation = ""

        # If the episode is done then
        if done:

            # if the vehicle just stopped in the middle of the road then subtract a penalty
            if stopped:
                reward -= 0.5
                explanation = "Vehicle stopped. "

            # Else if it crashed then subtract a penalty
            elif crash:
                reward -= 1.0
                explanation = "Crashed"

            # Else (episode ended successfully)
            else:
                # Add amount inversely proportional to the length of the episode
                reward = max(1.5 - 0.003 * self.steps_since_reset, 0.0)
                explanation = "Successful episode! {} steps".format(self.steps_since_reset)

        # Else, episode still underway
        else:

            # Penalty for speed exceeding the upper speed limit or for
            # going significantly slower than the speed limit
            norm_speed = self.obs[self.EGO_SPEED] * Car.MAX_SPEED / Car.ROAD_SPEED_LIMIT #1.0 = speed limit
            penalty = 0.0
            if norm_speed < 0.95:
                penalty = 0.02 * (1.0 - norm_speed/0.95)
                explanation += "Low speed penalty {:.4f}. ".format(penalty)
            elif norm_speed > 1.0:
                penalty = 0.03 * norm_speed - 0.03
                explanation += "HIGH speed penalty {:.4f}. ".format(penalty)
            reward -= penalty

        reward = min(max(reward, -2.0), 2.0)

        return reward, explanation
