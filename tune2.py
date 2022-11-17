import ray
from ray import air, tune
import ray.rllib.algorithms.ppo as ppo
#import ray.rllib.algorithms.a2c as a2c
#import ray.rllib.algorithms.sac as sac
#import ray.rllib.algorithms.ddpg as ddpg

from stopper2 import StopLogic
from car import Car

ray.init()

# Define which learning algorithm we will use
algo = "PPO"
params = ppo.DEFAULT_CONFIG.copy() #a2c requires empty dict

# Define the custom environment for Ray
env_config = {}
env_config["time_step_size"]                = 0.5
env_config["debug"]                         = 0
env_config["training"]                      = True
#env_config["action_clip_init"]              = tune.uniform(0.1, 1.0)
#env_config["action_clip_timesteps"]         = 100000

# Algorithm configs
params["env"]                               = Car
params["env_config"]                        = env_config
params["framework"]                         = "torch"
params["num_gpus"]                          = 1 #for the local worker
params["num_cpus_per_worker"]               = 1 #also applies to the local worker and evaluation workers
params["num_gpus_per_worker"]               = 0 #this has to allow for evaluation workers also
params["num_workers"]                       = 4 #num remote workers (remember that there is a local worker also)
params["num_envs_per_worker"]               = 1
params["rollout_fragment_length"]           = 200 #timesteps pulled from a sampler
params["gamma"]                             = 0.999 #tune.choice([0.99, 0.999])
params["evaluation_interval"]               = 6
params["evaluation_duration"]               = 6
params["evaluation_duration_unit"]          = "episodes"
params["evaluation_parallel_to_training"]   = True #True requires evaluation_num_workers > 0
params["evaluation_num_workers"]            = 2
params["log_level"]                         = "WARN"
params["seed"]                              = 17
params["batch_mode"]                        = "complete_episodes"

# ===== Params for DDPG =====================================================================
"""
explore_config = params["exploration_config"]
explore_config["type"]                      = "GaussianNoise" #default OrnsteinUhlenbeckNoise doesn't work well here
explore_config["stddev"]                    = 0.2 #tune.uniform(0.1, 0.5) #this param is specific to GaussianNoise
explore_config["random_timesteps"]          = 0 #tune.qrandint(0, 20000, 50000) #was 20000
explore_config["initial_scale"]             = 1.0
explore_config["final_scale"]               = 1.0
explore_config["scale_timesteps"]           = 1  #tune.choice([100000, 400000]) #was 900k
explore_config.pop("ou_base_scale")         #need to remove since this is specific to OU noise
explore_config.pop("ou_theta")              #need to remove since this is specific to OU noise
explore_config.pop("ou_sigma")              #need to remove since this is specific to OU noise

rb_config = params["replay_buffer_config"]
rb_config["capacity"]                       = 1000000

params["explore"]                           = True
params["exploration_config"]                = explore_config
params["replay_buffer_config"]              = rb_config
params["actor_hidden_activation"]           = "relu" #tune.choice(["tanh", "relu"])
params["critic_hidden_activation"]          = "relu" #tune.choice(["tanh", "relu"])
params["actor_hiddens"]                     = [512, 64] #[512, 128, 32]
params["critic_hiddens"]                    = [512, 64]
params["actor_lr"]                          = tune.loguniform(4e-8, 1e-4) #tune.choice([1e-5, 3e-5, 1e-4, 3e-4, 1e-3])
params["critic_lr"]                         = tune.loguniform(1e-6, 1e-3) #tune.loguniform(3e-5, 2e-4)
params["tau"]                               = 0.001 #tune.loguniform(0.00001, 0.003) #tune.loguniform(0.0005, 0.002)
params["train_batch_size"]                  = 32
"""
# ===== Params for PPO ======================================================================

params["lr"]                                = tune.loguniform(1e-7, 3e-4)
params["sgd_minibatch_size"]                = 32 #must be <= train_batch_size (and divide into it)
params["train_batch_size"]                  = 800 #must be = rollout_fragment_length * num_workers * num_envs_per_worker
#params["grad_clip"]                         = tune.uniform(0.1, 0.5)
#params["clip_param"]                        = None #tune.choice([0.2, 0.3, 0.6, 1.0])

# Add dict here for lots of model HPs
model_config = params["model"]
model_config["fcnet_hiddens"]               = tune.choice([[64, 48, 8], [50, 8], [16, 4]])
model_config["fcnet_activation"]            = "relu" #tune.choice(["relu", "relu", "tanh"])
model_config["post_fcnet_activation"]       = "linear" #tune.choice(["linear", "tanh"])
params["model"] = model_config

explore_config = params["exploration_config"]
explore_config["type"]                      = "GaussianNoise" #default OrnsteinUhlenbeckNoise doesn't work well here
explore_config["stddev"]                    = 0.2 #tune.uniform(0.1, 0.5) #this param is specific to GaussianNoise
explore_config["random_timesteps"]          = 0 #tune.qrandint(0, 20000, 50000) #was 20000
explore_config["initial_scale"]             = 1.0
explore_config["final_scale"]               = 0.02 #tune.choice([1.0, 0.01])
explore_config["scale_timesteps"]           = 500000  #tune.choice([100000, 400000]) #was 900k
params["exploration_config"] = explore_config

# ===== Final setup =========================================================================

print("\n///// {} training params are:\n".format(algo))
for item in params:
    print("{}:  {}".format(item, params[item]))

tune_config = tune.TuneConfig(
                metric                      = "episode_reward_mean",
                mode                        = "max",
                num_samples                 = 15 #number of HP trials
              )
stopper = StopLogic(max_timesteps           = 400,
                    max_iterations          = 800,
                    min_iterations          = 300,
                    avg_over_latest         = 200,
                    success_threshold       = 8.0,
                    failure_threshold       = 0.0,
                    compl_std_dev           = 0.1
                   )
run_config = air.RunConfig(
                name                        = "car",
                local_dir                   = "~/ray_results",
                stop                        = stopper,
                sync_config                 = tune.SyncConfig(syncer = None), #for single-node or shared checkpoint dir
                checkpoint_config           = air.CheckpointConfig(
                                                checkpoint_frequency        = 20,
                                                checkpoint_score_attribute  = "episode_reward_mean",
                                                num_to_keep                 = 3,
                                                checkpoint_at_end           = True
                )
             )

#checkpoint criteria: checkpoint_config=air.CheckpointConfig()

tuner = tune.Tuner(algo, param_space=params, tune_config=tune_config, run_config=run_config)
print("\n///// Tuner created.\n")

result = tuner.fit()
