import numpy as np
import tensorflow as tf
from collections import deque
import gym
import random
import copy
import math



class DQN:
    def __init__(self):
        self.ss = tf.Session()
        # placeholder: states, action, y-values
        self.inputStates = tf.placeholder("float", [None, 4])
        self.actions = tf.placeholder("float", [None, 2])
        self.yValues = tf.placeholder("float", [None]) # scalar Q value, single dimension, watch out


        # network, weights, bias
        # I=4, H1=100,  O=2

        INITIALIZATION_MEAN = 0.00
        INITIALIZATION_BIAS = -0.001

        _activation = lambda x : tf.maximum(0.01*x, x)

        self.W_IH1 = tf.Variable(tf.truncated_normal([4, 100], mean=INITIALIZATION_MEAN, stddev=0.1))
        self.B_H1 = tf.Variable(tf.constant(INITIALIZATION_BIAS, shape=[100]))
        self.H1 = tf.nn.relu((tf.matmul(self.inputStates, self.W_IH1)  + self.B_H1))

        self.W_H1H2 = tf.Variable(tf.truncated_normal([100, 100], mean=INITIALIZATION_MEAN, stddev=0.1))
        self.B_H2 = tf.Variable(tf.constant(INITIALIZATION_BIAS, shape=[100]))
        self.H2 = tf.nn.relu((tf.matmul(self.H1, self.W_H1H2)  + self.B_H2))

        self.W_H2H3 = tf.Variable(tf.truncated_normal([100, 100], mean=INITIALIZATION_MEAN, stddev=0.1))
        self.B_H3 = tf.Variable(tf.constant(INITIALIZATION_BIAS, shape=[100]))
        self.H3 = tf.nn.relu((tf.matmul(self.H2, self.W_H2H3)  + self.B_H3))

        self.W_H3O = tf.Variable(tf.truncated_normal([100, 2],  mean=INITIALIZATION_MEAN , stddev=0.1))
        self.B_O = tf.Variable(tf.constant(INITIALIZATION_BIAS, shape=[2]))
        self.O = tf.matmul(self.H3, self.W_H3O) + self.B_O

        # minimize the loss
        self.QValue = tf.reduce_sum(tf.mul(self.O, self.actions), reduction_indices=1) # [None], dimension reduction!
        self.loss = tf.reduce_sum(tf.square(self.QValue - self.yValues)) # collapse
        self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.loss) # this is the key, without this, you will get infinite values, wtf!

        self.ss.run(tf.initialize_all_variables())

    def predict(self,  states_tplus1):
        ret = self.ss.run(self.O, feed_dict={self.inputStates: states_tplus1}) # we are at time t+1, a basis for figuring out time=t case
        return ret

    def fit(self,  states_t, actions_t, yvalues):
        qvalue, loss, _ = self.ss.run([self.QValue, self.loss, self.optimizer],
                         feed_dict={self.inputStates: states_t,  # note the difference with the above func, we are timestamp t
                                          self.actions: actions_t,
                                          self.yValues: yvalues
                                          })
        # print(qvalue)



#special compared to cartpole-v0

def removePoor(replayBuffer):
    quarter = int(len(replayBuffer)/4.0)
    index = random.randint(0, quarter)
    replayBuffer.pop(index)

def updateReplayBuffer(replayBuffer, state, action, reward, newstate, terminal, i):


   if len(replayBuffer) >= bufferSize: # we need to get rid of poor guys periodically
        removePoor(replayBuffer)



   action_one_hot = np.zeros(env.action_space.n)
   action_one_hot[action]=1

   tuple = (state, action_one_hot, reward, newstate, terminal)
   insertPoint = 0
   for x in xrange(len(replayBuffer)):
       if tuple[2] < replayBuffer[x][2]:
            insertPoint = x
            break

   replayBuffer.insert(insertPoint, tuple)


# ok, let us get started!
if __name__=="__main__":


    env = gym.make('CartPole-v1')
    env.monitor.start('./cartpole-experiment-1', force=True)

    epsilon = 1.0
    decayRatio = 0.85
    gamma = 0.99 # why????

    replayBuffer = []
    bufferSize = 200000
    BATCH = 25
    ObservationMin = 50 # assert: observation_min at least should be larger than batch, providing sufficient data for net training

    episodes = 5000


    dqn = DQN()

    for i in xrange(episodes): # !!!!! do not forget the episodes!!!
        state = env.reset()

        total_reward = 0

        epsilon = max(epsilon * decayRatio, 0.1)


        while True:
            # select action
            # step = step + 1
            if i <= ObservationMin or random.random() <= epsilon:
                action = env.action_space.sample()
            else:
                twodstate = state.reshape(1, len(state))
                # outs = dqn.getOuts( twodstate)
                outs = dqn.predict(twodstate)
                action = np.argmax(outs[0])

            # perform action, update state
            newstate, reward, terminal, _ = env.step(action)
            total_reward += reward

            updateReplayBuffer(replayBuffer, state, action, reward, newstate, terminal, i)


            if i > ObservationMin:
                # dqn.train(replayBuffer, BATCH)

                #special compared to cartpole-v0
                sampleResult = random.sample(replayBuffer, 8*BATCH)
                sorted(sampleResult, key=lambda tup: tup[2], reverse=True)
                sampleResult = sampleResult[:len(sampleResult)/8]

                states_t = np.array([data[0] for data in sampleResult])
                actions_t = np.array([data[1] for data in sampleResult])
                rewards_t = np.array([data[2] for data in sampleResult])
                states_tplus1 = np.array([data[3] for data in sampleResult])
                nonterminal_t = np.array([1.-data[4] for data in sampleResult])
                outs_tplus1 = dqn.predict( states_tplus1)

                # feed the states[t], actions[t], yvalues[t] to the net to do the training.
                # yvalue[t] = reward[t]+ gamma * max(O), note the time stamp!
                yvalues_t = rewards_t.copy()
                yvalues_t = yvalues_t + gamma * outs_tplus1.max(axis=1) * nonterminal_t # if terminal, what is the point of the next state??
                dqn.fit( states_t, actions_t, yvalues_t)


            state = newstate.copy()

            if terminal: # let us move to the next episode
                if i > ObservationMin:
                    print ('Episode', i, 'Reward', total_reward)

                break


    env.monitor.close()
