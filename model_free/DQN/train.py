import gym
from agent import Agent
from config import Config

def main(cf):
    
    max_frames = 10000000
    
    env = gym.make(cf.atari_games[0])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    ag = Agent(env, state_dim, action_dim, cf)
    ag.train(max_frames)

if __name__ == "__main__":
    cf = Config()
    main(cf)