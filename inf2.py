from cmath import inf
import sys
import math
import numpy as np
import gym
import ray
import ray.rllib.algorithms.ppo as ppo
#import ray.rllib.algorithms.ddpg as ddpg
from car import Car

"""This program runs the selected policy checkpoint for one episode and captures key state variables throughout."""

def main(argv):

    prng = np.random.default_rng()

    # Handle any args
    if len(argv) == 1:
        print("Usage is: {} <checkpoint>".format(argv[0]))
        sys.exit(1)

    checkpoint = argv[1]
    start_lane = 0          #for forward compatibility

    ray.init()

    # Set up the environment
    config = ppo.DEFAULT_CONFIG.copy()

    env_config = {  "time_step_size":   0.5,
                    "debug":            0,
                    "init_ego_lane":    start_lane
                }

    # DDPG - These need to be same as in the checkpoint being read!
    """
    exp = config["exploration_config"]
    exp["type"] = "GaussianNoise"
    exp.pop("ou_sigma")
    exp.pop("ou_theta")
    exp.pop("ou_base_scale")
    config["exploration_config"] = exp
    config["actor_hiddens"]               = [512, 64]
    config["critic_hiddens"]              = [512, 64]
    """

    # PPO - need to match checkpoint being read!
    model = config["model"]
    model["fcnet_hiddens"]          = [64, 48, 8]
    #model["fcnet_hiddens"]          = [1024, 128, 16]
    model["fcnet_activation"]       = "relu"
    model["post_fcnet_activation"]  = "linear"
    config["model"] = model

    config["env_config"] = env_config
    config["framework"] = "torch"
    config["num_gpus"] = 0
    config["num_workers"] = 1
    config["seed"] = 17
    env = Car(env_config)

    # Restore the selected checkpoint file
    # Note that the raw environment class is passed to the algo, but we are only using the algo to run the NN model,
    # not to run the environment, so any environment info we pass to the algo is irrelevant for this program.  The
    # algo doesn't recognize the config key "env_configs", so need to remove it here.
    #config.pop("env_configs")
    algo = ppo.PPO(config = config, env = Car) #needs the env class, not the object created above
    algo.restore(checkpoint)
    print("///// Checkpoint {} successfully loaded.".format(checkpoint))

    # Run the agent through a complete episode
    episode_reward = 0
    done = False
    obs = env.reset()
    step = 0
    while not done:
        step += 1
        action = algo.compute_single_action(obs)
        obs, reward, done, info = env.step(np.ndarray.tolist(action)) #obs returned is already scaled
        episode_reward += reward

        # Display current status of the ego vehicle
        raw_obs = obs.copy()
        raw_obs[0] *= 2000.0
        print("///// step {:3d}: scaled action = [{:7.4f}], speed = {:.3f}, dist = {:.3f}, r = {:7.4f} {}"
                .format(step, action[0], raw_obs[1], raw_obs[0], reward, info["reward_detail"]))

        if done:
            print("///// Episode complete: {}. Total reward = {:.2f}".format(info["reason"], episode_reward))


######################################################################################################
######################################################################################################


if __name__ == "__main__":
   main(sys.argv)
