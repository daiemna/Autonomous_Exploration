# import numpy.random as random
import random
import numpy
import os.path as osp
import ast
import pprint

from keras.models import Model
from keras.layers import Convolution2D, Dense, Flatten, Input, MaxPooling2D
from keras.engine.topology import Merge
from keras.optimizers import RMSprop
from keras import backend as K
from keras.utils.visualize_util import plot
from theano import printing
from theano.gradient import disconnected_grad

class DeepQAgent:
    def __init__(self, state_size, number_of_actions, learning_rate=0.01,
                 epsilon=0.1, mbsz=32, discount=0.9, memory=50,state_file="dqn_state.json",
                 cnn_save_name='CNN', save_freq=10, random_episodes=500,
                 cnn_layers=[[16,8,4,'relu'],[32,4,2,'relu'],[256,'relu']],
                 nn_layers=[[256,'relu']]):
        if isinstance(state_size,list)  and len(state_size) == 2 and \
           isinstance(state_size[0], tuple) and \
           isinstance(state_size[1], tuple):
            self.cnn_input_size = state_size[0]
            self.nn_input_size = state_size[1]
            nn_name = ["NN" , str(state_size[1][0])]
            [nn_name.append("x".join(map(str,l))) for l in nn_layers]
            cnn_save_name = "_".join(nn_name) + "_" + cnn_save_name

        elif isinstance(state_size, tuple):
            self.cnn_input_size = state_size
            self.nn_input_size = None
        else:
            raise ValueError("state_size can be a tuple or a list in case of CNN or CNN + NN respectivly.")


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
        if not nn_layers or self.nn_input_size is None:
            self.log["info"]("Not building MLP!")
            self.is_building_mlp = False
        else:
            self.nn_layers = nn_layers
            self.is_building_mlp = True

        if osp.exists(self.state_file):
            with open(self.state_file, "r") as js:
                self.state = ast.literal_eval(js.read())
        else:
            self.state = {}
        print "state is : ", str(self.state)
        self.i = self.state.get("i", 1);

        self.save_freq = save_freq
        self.cnn_layers=cnn_layers
        self.cnn_save_name = [cnn_save_name, "x".join(map(str,self.cnn_input_size))]
        [self.cnn_save_name.append("x".join(map(str,l))) for l in cnn_layers]
        self.cnn_save_name = "_".join(self.cnn_save_name)

        self.is_last_action_random = False
        self.build_functions()

    def build_cnn_model(self):
        if self.is_building_mlp:
            Sn = Input(shape=self.nn_input_size)
            for i,layer in enumerate(self.nn_layers):
                if not len(layer) == 2:
                    raise ValueError("NN layers must be list of 2, containing number preceptrons and activation")
                if i == 0:
                    h = Dense(layer[0], activation=layer[1])(Sn)
                else:
                    h = Dense(layer[0], activation=layer[1])(h)
            nn_model_=h

        S = Input(shape=self.cnn_input_size)
        for i,layer in enumerate(self.cnn_layers):
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
                if self.is_building_mlp:
                    h = Merge(mode='concat')([h, nn_model_])
                h = Dense(layer[0], activation=layer[1])(h)
            else:
                raise ValueError("cnn_layers can have 4 parameters for Convolutional or 2 parameters for Dense layer. given %d"%len(layer))
        V = Dense(self.number_of_actions)(h)

        if self.is_building_mlp:
            self.model = Model([S,Sn], V)
        else:
            self.model = Model(S, V)
        plot(self.model, to_file='{}.png'.format(self.cnn_save_name))
        try:
            self.model.load_weights('{}.h5'.format(self.cnn_save_name))
            print "loading from {}.h5".format(self.cnn_save_name)
        except:
            print "Training a new model"

        # def build_mlp_model(self):
        # raise NotImplemented("did not do it, I am lazy!")
        # Sn = Input(shape=self.nn_input_size)
        # for i,layer in enumerate(self.nn_layers):
        #     if not len(layer) == 2:
        #         raise ValueError("NN layers must be list of 2, containing number preceptrons and activation")
        #     if i == 0:
        #         h = Dense(layer[0], activation=layer[1])(Sn)
        #     else:
        #         h = Dense(layer[0], activation=layer[1])(h)
        # self.nn_model_=h


    def build_functions(self):
        A = Input(shape=(1,), dtype='int32')
        R = Input(shape=(1,), dtype='float32')
        T = Input(shape=(1,), dtype='int32')

        if self.is_building_mlp:
            CNN_State = Input(shape=self.cnn_input_size)
            NN_State = Input(shape=self.nn_input_size)
            State = [CNN_State, NN_State]

            CNN_NState = Input(shape=self.cnn_input_size)
            NN_NState = Input(shape=self.nn_input_size)
            NState = [CNN_NState, NN_NState]
        else:
            State = Input(shape=self.cnn_input_size)
            NState = Input(shape=self.cnn_input_size)
        self.log["debug"]("State : " + str(State))
        self.log["debug"]("NState : " + str(NState))
        self.build_cnn_model()
        if self.is_building_mlp:
            self.value_fn = K.function(State, self.model(State))
            VS = self.model(State)
            VNS = disconnected_grad(self.model(NState))
        else:
            self.value_fn = K.function([State], self.model(State))
            VS = self.model([State])
            VNS = disconnected_grad(self.model([NState]))
        future_value = (1-T) * VNS.max(axis=1, keepdims=True)
        discounted_future_value = self.discount * future_value
        target = R + discounted_future_value
        cost = ((VS[:, A] - target)**2).mean()
        opt = RMSprop(lr=self.lr)
        params = self.model.trainable_weights
        updates = opt.get_updates(params, [], cost)
        if self.is_building_mlp:
            self.train_fn = K.function([CNN_State, NN_State, CNN_NState, NN_NState, A, R, T], cost, updates=updates)
        else:
            self.train_fn = K.function([State, NState, A, R, T], cost, updates=updates)


    def new_episode(self):
        self.states.append([])
        self.actions.append([])
        self.rewards.append([])
        self.states = self.states[-self.memory:]
        self.actions = self.actions[-self.memory:]
        self.rewards = self.rewards[-self.memory:]
        self.i += 1
        if self.i % self.save_freq == 0:
            self.model.save_weights('{}.h5'.format(self.cnn_save_name), True)
        self.change_key_save("i", self.i)

    def end_episode(self, terminal_state):
        self.states[-1].append(terminal_state)

    def change_key_save(self, key,value):
        self.state[key] = value
        with open(self.state_file, "w") as js:
            js.write(str(self.state))

    def act(self, state):
        self.log["debug"]("State 1: " + str(state[0].shape))
        self.log["debug"]("State 2: " + str(state[1].shape))
        self.states[-1].append(state)
        values = self.value_fn(state)
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
        if not self.is_building_mlp:
            S = numpy.zeros((self.mbsz,) + self.cnn_input_size)
            NS = numpy.zeros((self.mbsz,) + self.cnn_input_size)
        else:
            S = numpy.zeros((self.mbsz,) + self.cnn_input_size)
            NS = numpy.zeros((self.mbsz,) + self.cnn_input_size)
            Sn = numpy.zeros((self.mbsz,) + self.nn_input_size)
            NSn = numpy.zeros((self.mbsz,) + self.nn_input_size)

        A = numpy.zeros((self.mbsz, 1), dtype=numpy.int32)
        R = numpy.zeros((self.mbsz, 1), dtype=numpy.float32)
        T = numpy.zeros((self.mbsz, 1), dtype=numpy.int32)

        for i in xrange(self.mbsz):
            episode = random.randint(max(0, N-self.memory), N-1)
            num_frames = len(self.states[episode])
            self.log["debug"]("number of Frames : "+ str(num_frames))
            frame = random.randint(0, num_frames-2)
            self.log["debug"]("Frame : "+ str(frame))
            if not self.is_building_mlp:
                S[i] = self.states[episode][frame]
                if frame < num_frames - 1:
                    NS[i] = self.states[episode][frame+1]
            else:
                S[i] = self.states[episode][frame][0]
                Sn[i] = self.states[episode][frame][1]
                if frame < num_frames - 1:
                    NS[i] = self.states[episode][frame+1][0]
                    NSn[i] = self.states[episode][frame+1][1]
            T[i] = 1 if frame == num_frames - 1 else 0
            A[i] = self.actions[episode][frame]
            R[i] = self.rewards[episode][frame]
        if not self.is_building_mlp:
            cost = self.train_fn([S, NS, A, R, T])
        else:
            cost = self.train_fn([S, Sn, NS, NSn, A, R, T])
        self.log["debug"]("Cost incurred : "+ str(cost))
        return cost
