# import numpy.random as random
import random
import numpy
import os.path as osp
import ast
import pprint

from keras.models import Model
from keras.layers import Convolution2D, Dense, Flatten, Input, merge, MaxPooling2D
from keras.optimizers import RMSprop
from keras import backend as K
from theano import printing
from theano.gradient import disconnected_grad

class DeepQAgent:
    def __init__(self, state_size=None, number_of_actions=1,learning_rate=0.01,
                 epsilon=0.1, mbsz=32, discount=0.9, memory=50,state_file="dqn_state.json",
                 save_name='CNN', save_freq=10, random_episodes=500,
                 layers=[[16,8,4,'relu'],[32,4,2,'relu'],[256,'relu']]):
        self.state_size = state_size
        self.number_of_actions = number_of_actions
        self.epsilon = epsilon
        self.random_episodes = random_episodes
        self.mbsz = mbsz
        self.discount = discount
        self.lr = learning_rate
        self.memory = memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.experience = []
        self.state_file = state_file
        self.log = {
            "debug":pprint.pprint,
            "info":pprint.pprint
        }
        # self.logdeinfo=print

        if osp.exists(self.state_file):
            with open(self.state_file, "r") as js:
                self.state = ast.literal_eval(js.read())
        else:
            self.state = {}
        print "state is : ", str(self.state)
        self.i = self.state.get("i", 1);

        self.save_freq = save_freq
        self.layers=layers
        self.save_name = [save_name, "x".join(map(str,state_size))]
        [self.save_name.append("x".join(map(str,l))) for l in layers]
        self.save_name = "_".join(self.save_name)

        self.is_last_action_random = False
        self.build_functions()

    def build_model(self):
        S = Input(shape=self.state_size)
        for i,layer in enumerate(self.layers):
            if len(layer) == 4:
                if i == 0:
                    h = Convolution2D(layer[0], layer[1], layer[1],
                                      subsample=(layer[2], layer[2]),
                                      border_mode='same', activation=layer[3])(S)
                else:
                    h = Convolution2D(layer[0], layer[1], layer[1],
                                      subsample=(layer[2], layer[2]),
                                      border_mode='same', activation=layer[3])(h)
                h = MaxPooling2D(pool_size=(2, 2))(h)
            elif len(layer) == 2:
                h = Flatten()(h)
                h = Dense(layer[0], activation=layer[1])(h)
            else:
                raise ValueError("Layers can have 4 parameters for Convolutional or 2 parameters for Dense layer. given %d"%len(layers))
        V = Dense(self.number_of_actions)(h)
        self.model = Model(S, V)
        try:
            self.model.load_weights('{}.h5'.format(self.save_name))
            print "loading from {}.h5".format(self.save_name)
        except:
            print "Training a new model"


    def build_functions(self):
        S = Input(shape=self.state_size)
        NS = Input(shape=self.state_size)
        A = Input(shape=(1,), dtype='int32')
        R = Input(shape=(1,), dtype='float32')
        T = Input(shape=(1,), dtype='int32')
        self.build_model()
        self.value_fn = K.function([S], self.model(S))

        VS = self.model(S)
        VNS = disconnected_grad(self.model(NS))
        future_value = (1-T) * VNS.max(axis=1, keepdims=True)
        discounted_future_value = self.discount * future_value
        target = R + discounted_future_value
        cost = ((VS[:, A] - target)**2).mean()
        opt = RMSprop(lr=self.lr)
        params = self.model.trainable_weights
        updates = opt.get_updates(params, [], cost)
        self.train_fn = K.function([S, NS, A, R, T], cost, updates=updates)

    def new_episode(self):
        self.states.append([])
        self.actions.append([])
        self.rewards.append([])
        self.states = self.states[-self.memory:]
        self.actions = self.actions[-self.memory:]
        self.rewards = self.rewards[-self.memory:]
        self.i += 1
        if self.i % self.save_freq == 0:
            self.model.save_weights('{}.h5'.format(self.save_name), True)
        self.change_key_save("i", self.i)

    def end_episode(self, terminal_state):
        self.states[-1].append(terminal_state)

    def change_key_save(self, key,value):
        self.state[key] = value
        with open(self.state_file, "w") as js:
            js.write(str(self.state))

    def act(self, state):
        self.states[-1].append(state)
        values = self.value_fn([state[None, :]])
        if numpy.random.random() < self.epsilon and self.i <= self.random_episodes:
            action = numpy.random.randint(self.number_of_actions)
            self.is_last_action_random = True
        else:
            self.is_last_action_random = False
            action = values.argmax()
        self.actions[-1].append(action)
        return action, values

    def observe(self, reward):
        self.log["debug"]("Reward : " + str(reward))
        self.rewards[-1].append(reward)
        return self.iterate()

    def iterate(self):
        N = len(self.states)
        if N < self.mbsz:
            return 0
        S = numpy.zeros((self.mbsz,) + self.state_size)
        NS = numpy.zeros((self.mbsz,) + self.state_size)
        A = numpy.zeros((self.mbsz, 1), dtype=numpy.int32)
        R = numpy.zeros((self.mbsz, 1), dtype=numpy.float32)
        T = numpy.zeros((self.mbsz, 1), dtype=numpy.int32)
        for i in xrange(self.mbsz):
            episode = random.randint(max(0, N-self.memory), N-1)
            num_frames = len(self.states[episode])
            self.log["debug"]("number of Frames : "+ str(num_frames))
            frame = random.randint(0, num_frames-2)
            self.log["debug"]("Frame : "+ str(frame))
            S[i] = self.states[episode][frame]
            T[i] = 1 if frame == num_frames - 1 else 0
            if frame < num_frames - 1:
                NS[i] = self.states[episode][frame+1]
            A[i] = self.actions[episode][frame]
            R[i] = self.rewards[episode][frame]
        cost = self.train_fn([S, NS, A, R, T])
        self.log["debug"]("Cost incurred : "+ str(cost))
        return cost
