import gym
from agent import Agent
from config import Config

def main(cf):
    
    max_frames = 10000000
    
    env = gym.make(cf.atari_games[0])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    ag = Agent(cf, env, state_dim, action_dim)
    ag.run(max_frames, cf.atari_games[0], render=False)

if __name__ == "__main__":
    cf = Config()
    main(cf)