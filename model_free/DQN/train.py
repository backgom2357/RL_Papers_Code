import gym
from agent import Agent
from config import Config
import os
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def main(cf):
    
    max_frames = 10000000
    game = cf.ATARI_GAMES[6]
    env = gym.make(game)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    ag = Agent(cf, env, state_dim, action_dim)
    ag.run(max_frames, game, render=False)

if __name__ == "__main__":
    cf = Config()
    main(cf)