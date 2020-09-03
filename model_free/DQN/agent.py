from network import *
from replay_memory import ReplayMemory
from utils import *
import numpy as np
import cv2
import tensorflow as tf
import os
# import wandb

class Agent:
    def __init__(self, config, env, state_dim, action_dim):
        
        # Get  Config
        self.cf = config

        # Setting Environment
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Setting Replay Memory
        self.rm = ReplayMemory(self.cf.replay_memory_size, self.cf.frame_size, self.cf.agent_history_length)
        
        # Build Model
        self.q = build_model(self.cf.frame_size, self.action_dim, self.cf.agent_history_length)
        self.target_q = build_model(self.cf.frame_size, self.action_dim, self.cf.agent_history_length)

        # Optimizer and Loss for Training
        self.optimizer=tf.keras.optimizers.Adam(learning_rate=self.cf.learning_rate, clipnorm=10.)
        self.loss = tf.keras.losses.Huber()
        self.q.summary()

    def get_action(self, state):
        """
        Epsilon Greedy
        """
        q = self.q(state)[0]
        return np.argmax(q), q if self.cf.epsilon < np.random.rand() else np.random.randint(self.action_dim), q

    def model_train(self):
        # Sample From Replay Memory
        states, actions, rewards, next_states, dones = self.replay_memory.sample(self.cf.batch_size)

        # Epsilon Decay
        if self.cf.epsilon > self.cf.final_exploration:
            self.cf.epsilon -= (self.cf.initial_exploration + self.cf.final_exploration)/(self.cf.final_exploration_frame*self.cf.epsilon)
        
        # Update Weights
        with tf.GradientTape() as g:
            # Maximum q value of next state from target q function
            max_next_q = np.max(self.target_q(next_states), axis=1)

            # Calculate Targets
            targets = rewards + (1 - dones) * (self.cf.discount_factor * max_next_q)
            predicts = self.q(states)
            loss = self.loss(targets, predicts)
        g_theta = g.gradient(loss, self.q.trainable_weights)
        self.optimizer.apply_gradients(g_theta, self.q.trainable_weights)


    def run(self, max_frame, game_name, render=False):
        
        # For the Logs
        sum_mean_q, episodic_rewards = 0, 0

        # Initalizing
        episode = 0
        frames, action = 0, 0
        initial_state = self.evn.reset()
        state = np.stack([preprocess(initial_state, frame_size=self.cf.frame_size)]*4, axis=3)
        state = np.reshape(state, (1, *state.shape))

        # No Ops
        for _ in range(self.cf.no_ops):
            next_state, _, _,  _ = self.env.step(0)
            next_state = np.append(state[..., 1:], preprocess(next_state, frame_size=self.cf.frame_size), axis=3))
            state = next_state

        while frames < max_frame:

            if render:
                self.env.render()

            # Interact with Environmnet
            action, q = self.get_action(normalize(state))
            state, reward, done, _ = self.env.step(action)
            reward = np.clip(reward, -1, 1)
            next_state = np.append(state[..., 1:], preprocess(next_state, frame_size=self.cf.frame_size), axis=3))

            # Append To Replay Memeory
            self.replay_memory.append(state, action, reward, next_state, done)

            # Start Training After Collecting Enough Samples
            if self.relay_memeory.crt < self.cf.replay_start_size and not self.replay_memory.is_full():
                state = next_state
                continue
            
            # Training
            self.model_train()
            state = next_state

            episodic_rewards += reward
            sum_mean_q += np.mean(q)

            frames += 1

            # Update Target Q
            if frames % self.cf.target_q_update_frequency  == 0:
                self.target_q.set_weights(self.q.get_weights())

            if done:
                episodic_mean_q = sum_mean_q/frames(self.cf.skip_frames+1)
                episode += 1

                # Update Logs
                print(f'Epi : {self.episode,}, Reward : {episodic_rewards}, Q : {episodic_mean_q}')
                wandb.log({
                    'Reward':episode_reward, 
                    'Q value':episodic_mean_q, 
                    })

                # Save Model
                if episode % 1000 == 0:
                    self.q.save_weights('./save_weights/'+game_name+'_'+str(self.episode)+'epi.h5')

                # Initializing
                sum_mean_q, episodic_rewards = 0, 0
                action = 0
                initial_state = self.evn.reset()
                state = np.stack([preprocess(initial_state, frame_size=self.cf.frame_size)]*4, axis=3)
                state = np.reshape(state, (1, *state.shape))
                for _ in range(self.cf.no_ops):
                    next_state, _, _,  _ = self.env.step(0)
                    next_state = np.append(state[..., 1:], preprocess(next_state, frame_size=self.cf.frame_size), axis=3))
                    state = next_state

                
        
        