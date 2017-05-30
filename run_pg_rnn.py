import numpy as np
import json
import os
import gym
import tensorflow as tf
from tqdm import tqdm
from pg_rnn import PolicyGradientRNN
from sampler import Sampler
from model import policy_network

config = json.load(open("configuration.json"))
train = config["train"]

env = gym.make(config["env_name"])

observation_dim = env.observation_space.shape
num_actions = env.action_space.n

# RNN configuration
global_step = tf.Variable(0, name="global_step", trainable=False)
learning_adaptive = config["learning"]["learning_adaptive"]
if learning_adaptive:
    learning_rate = tf.train.exponential_decay(
                      config["learning"]["learning_rate"],
                      global_step,
                      config["learning"]["decay_steps"],
                      config["learning"]["decay_rate"],
                      staircase=True)
else:
    learning_rate = config["learning"]["learning_rate"]

#tensorflow
sess = tf.Session()
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

# checkpointing
base_file = "_".join([k + "-" + str(v) for k, v in sorted(config.items())
                                    if k not in ["train", "learning"]])
os.makedirs(base_file, exist_ok=True)
json.dump(config, open(base_file + "/configuration.json", "w"))
writer = tf.summary.FileWriter(base_file + "/summary/")
save_path = base_file + '/models/'
os.makedirs(save_path, exist_ok=True)

pg_rnn = PolicyGradientRNN(sess,
                           optimizer,
                           policy_network,
                           observation_dim,
                           num_actions,
                           config["gru_unit_size"],
                           config["num_step"],
                           config["num_layers"],
                           save_path + env.spec.id,
                           global_step,
                           config["max_gradient_norm"],
                           config["entropy_bonus"],
                           writer,
                           loss_function=config["loss_function"],
                           summary_every=10)

sampler = Sampler(pg_rnn,
                  env,
                  config["gru_unit_size"],
                  config["num_step"],
                  config["num_layers"],
                  config["max_step"],
                  config["batch_size"],
                  config["discount"],
                  writer)

reward = []
for _ in tqdm(range(config["num_itr"])):
    if train:
        batch = sampler.samples()
        pg_rnn.update_parameters(batch["observations"], batch["actions"],
                                batch["returns"], batch["init_states"],
                                batch["seq_len"])
    else:
        episode = sampler.collect_one_episode(render=True)
        print("reward is {0}".format(np.sum(episode["rewards"])))
