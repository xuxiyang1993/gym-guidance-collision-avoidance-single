import os
import numpy as np
import random
import time
from copy import copy
from collections import deque
from keras.models import Sequential
from keras.layers import Dense,Convolution2D,Flatten,Activation,LeakyReLU,Dropout
from keras.optimizers import Adam
from keras import backend as K
import argparse
import tensorflow as tf
import geopy.distance
from bluesky.tools import geo
from operator import itemgetter

################################
##                            ##
##      Marc Brittain         ##
##  marcbrittain.github.io    ##
##                            ##
################################


def dist_goal(states,traf,id):

    for i in range(len(traf.id)):
        if traf.id[i] == id:
            olat,olon = states[0],states[1]
            ilat,ilon =traf.ap.route[i].wplat[0],traf.ap.route[i].wplon[0]
            _,dist = geo.qdrdist(olat,olon,ilat,ilon)
            return dist





def getClosestAC_Distance(self,state,traf,route_keeper):

    olat,olon,ID = state[:3]
    rte = route_keeper[ID]
    dist = []
    #print(olat,olon)
    for i in range(len(traf.lat)):
        _,d = geo.qdrdist(olat,olon,traf.lat[i],traf.lon[i])
        dist.append([d,i])


    dist = sorted(dist,key=itemgetter(0))[1:]
     ##first entry will be ownship since distance is 0

    dist = self.getValidDistance(olat,olon,rte,dist,traf,route_keeper)


    return dist





# initalize the DDQN agent
class A2C_Agent:
    def __init__(self,state_size,action_size,num_routes,n_closest,numEpisodes,message_size,lag_values,positions,intersections):

        self.state_size = state_size + 1 + 3*n_closest
        self.action_size = action_size
        self.message_size=message_size
        self.positions = positions
        self.intersections = intersections
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0   # initial epsilon value
        self.numEpisodes = numEpisodes
        self.epsilon_vals = np.append(np.linspace(self.epsilon,0.1,1000),np.linspace(0.1,0.01,4000))
        self.beta_vals = np.linspace(0.4,1.0,self.numEpisodes)
        self.dropouts = np.linspace(0.9,0,2000)
        self.max_time = 500
        self.lag_values = lag_values

        self.episode_count = 0
        self._goals = []

        #if windows or linux
        self.speeds = np.array([157,0,347])

        #if mac
        #self.speeds = np.array([101,259])
        self.total_reward = 0
        self.init=True
        self.finished = False
        self.batch_size = 30
        self.num_routes = num_routes
        self.n_closest = n_closest
        self.experience = {}
        self.memory = deque(maxlen=30)

        #self.cas_max = 178.15464822
        #self.cas_min = 80.767708

        self.tas_max = 277.4740898435427
        self.tas_min = 132.2803067558585

        # IF mac
        #self.cas_max = 133.2409960000002
        #self.cas_min = 51.95884400000024

        #self.tas_max = 212.75640607975387
        #self.tas_min = 85.92365559447086


        self.lr = 0.0005

        self.value_size = 1

        self.local_copies = {}

        self.fit_count = 0

        self.getRouteDistances()

        #print(self.intersections)
        #print(self.positions)




        # flag is for running an episode with epsilon = 1e-10
        self.flag = False

        # saving our values from the flag episode
        self.model_check = []

        #self.learning_rate = 0.0005      #optimizer leanring rate



        self.model = self._build_A2C()

        # initialize model
        #self.sess = tf.Session()
        #self.sess.run(tf.initialize_all_variables())

        self.UPDATE_FREQ = 7000  # how often to update the target network

        self.count = 0

    def getValidDistance(self,lat,lon,rte,dist,traf,route_keeper):

        #closest_dist = np.inf

        for i in range(len(dist)):
            d,index = dist[i]

            rte_int = route_keeper[traf.id[index]]

            if rte == 0:
                if rte_int == 2:
                    if traf.lat[index] > self.intersections[0][0] and lon > self.intersections[0][1]:
                        return d

                if rte_int == 0:
                    return d

            if rte == 1:
                if rte_int == 2:
                    if traf.lat[index] > self.intersections[1][0] and lon < self.intersections[1][1]:
                        return d
                if rte_int == 1:
                    return d

            if rte == 2:
                if rte_int == 1:
                    if traf.lon[index] < self.intersections[1][1] and lat > self.intersections[1][0]:
                        return d

                if rte_int == 0:
                    if traf.lon[index] > self.intersections[0][1] and lat > self.intersections[0][0] and lat < self.intersections[1][0]:
                        return d

                if rte_int == 2:
                    return d

        return np.inf

    def getRouteDistances(self):
        self.intersection_distances = []
        self.route_distances = []
        for i in range(len(self.positions)):
            olat,olon,glat,glon,h = self.positions[i]
            _, d = geo.qdrdist(olat,-olon,glat,-glon)
            self.route_distances.append(d)

        for i in range(len(self.positions)):
            olat,olon,glat,glon,h = self.positions[i]
            if i == 0:
                Ilat,Ilon = self.intersections[0]
                _,d = geo.qdrdist(Ilat,Ilon,glat,-glon)
                self.intersection_distances.append(d)

            if i == 1:
                Ilat,Ilon = self.intersections[1]
                _,d = geo.qdrdist(Ilat,Ilon,glat,-glon)
                self.intersection_distances.append(d)


            if i == 2:
                d = []
                Ilat,Ilon = self.intersections[1]
                _,d_  = geo.qdrdist(Ilat,Ilon,glat,-glon)
                d.append(d_)
                Ilat,Ilon = self.intersections[0]
                _,d_  = geo.qdrdist(Ilat,Ilon,glat,-glon)
                d.append(d_)

                self.intersection_distances.append(d)





    def dist_intersection(self,lat,lon,route):
        if route == 0:
            Ilat,Ilon = self.intersections[0]

        if route == 1:
            Ilat,Ilon = self.intersections[1]

        if route == 2:
            if lat > self.intersections[1][0]:
                Ilat,Ilon = self.intersections[1]

            else:
                Ilat,Ilon = self.intersections[0]

        distance = geopy.distance.geodesic((lat,lon),(Ilat,Ilon)).m

        return distance

    def discount(self, r):
        """ Compute the gamma-discounted rewards over an episode
        """
        discounted_r, cumul_r = np.zeros_like(r), 0
        for t in reversed(range(0, len(r))):
            cumul_r = r[t] + cumul_r * self.gamma
            discounted_r[t] = cumul_r
        return discounted_r




    def normalize_that(self,value,what):

        if what=='latlon':
            lat,lon,route = value
            route = int(route)
            olat,olon,glat,glon,h = self.positions[route]

            max_d = max(self.route_distances)
            _,d  = geo.qdrdist(lat,lon,glat,-glon)

            return d/max_d

        if what=='spd':

            if value > self.tas_max:
                self.tas_max = value

            if value < self.tas_min:
                self.tas_min = value
            return (value-self.tas_min)/(self.tas_max-self.tas_min)
            #return value/self.cas_max

        if what=='rt':
            return value/2

        if what=='dist':
            lat, route = value
            route = int(route)

            if route == 0 or route == 1:
                return self.intersection_distances[route]/max(self.route_distances)

            if route == 2 and lat >= self.intersections[1][0]:
                return self.intersection_distances[route][0]/max(self.route_distances)

            if route ==2 and lat < self.intersections[1][0]:
                return self.intersection_distances[route][1]/max(self.route_distances)

        if what == 'acc':
            return value+0.5


        if what =='mess':
            return value/(self.message_size-1)

        if what=='state':


            norm_array = np.zeros((self.lag_values,(self.state_size)))

            for k in range(self.lag_values):

                norm_state = []

                dgoal = self.normalize_that((value[k][0],value[k][1],value[k][3]),'latlon')
                spd = self.normalize_that(value[k][2],'spd')
                rt = self.normalize_that(value[k][3],'rt')
                dinter_own = abs(self.normalize_that((value[k][0],value[k][3]),'dist') - dgoal)
                acc = self.normalize_that(value[k][4],'acc')

                rt_own = int(value[k][3])


                norm_state.append(dgoal)
                norm_state.append(spd)
                norm_state.append(rt)
                norm_state.append(dinter_own)
                norm_state.append(acc)
                norm_state.append(1.5/max(self.route_distances))


                for i in range(5,len(value[k]),5):

                    if not value[k][i] == 0:
                        dgoal = self.normalize_that((value[k][i],value[k][i+1],value[k][i+3]),'latlon')
                        spd = self.normalize_that(value[k][i+2],'spd')
                        rt = self.normalize_that(value[k][i+3],'rt')
                        dinter = abs(self.normalize_that((value[k][i],value[k][i+3]),'dist') - dgoal)
                        acc = self.normalize_that(value[k][i+4],'acc')

                        if rt_own == int(value[k][i+3]):
                            dist_away = abs(dinter_own-dinter)
                        else:
                            dist_away = np.sqrt(dinter_own**2 + dinter**2)

                        rt_int = int(value[k][i+3])

                        norm_state.append(dgoal)
                        norm_state.append(spd)
                        norm_state.append(rt)
                        norm_state.append(dinter)
                        norm_state.append(dist_away)
                        norm_state.append(acc)
                        norm_state.append(1.5/max(self.route_distances))
                        norm_state.append(1.5/max(self.route_distances) + 1.5/max(self.route_distances))



                    else:
                        for j in range(8):
                            norm_state.append(0)

                norm_array[k,:] = np.array(norm_state)




            return norm_array


    def _build_A2C(self):

        I = tf.keras.layers.Input(shape=(self.state_size,),name='states')

        # dgoal, dintersection, route, speed, acceleration, radius
        own_state = tf.keras.layers.Lambda(lambda x: x[:,:6],output_shape=(6,))(I)

        # dgoal, dintersection, route, speed, acceleration, radius, downship,LOS
        other_state = tf.keras.layers.Lambda(lambda x: x[:,6:],output_shape=(self.state_size-6,))(I)

        # encoding other_state into 32 values
        H1_int = tf.keras.layers.Dense(32,activation='relu')(other_state)


        # now combine them
        combined = tf.keras.layers.concatenate([own_state,H1_int], axis=-1)


        H2 = tf.keras.layers.Dense(256,activation='relu')(combined)
        H3 = tf.keras.layers.Dense(256,activation='relu')(H2)


        output = tf.keras.layers.Dense(self.action_size+1,activation=None)(H3)

        # Split the output layer into policy and value
        policy = tf.keras.layers.Lambda(lambda x: x[:,:self.action_size],output_shape=(self.action_size,))(output)
        value = tf.keras.layers.Lambda(lambda x: x[:,self.action_size:],output_shape=(self.value_size,))(output)

        # now I need to apply activation

        policy_out = tf.keras.layers.Activation('softmax',name='policy_out')(policy)
        value_out = tf.keras.layers.Activation('linear',name='value_out')(value)

        # Using Adam optimizer, RMSProp's successor.
        opt = tf.keras.optimizers.Adam(lr=self.lr)


        model = tf.keras.models.Model(inputs=I, outputs=[policy_out,value_out])

        # The model is trained on 2 different loss functions
        model.compile(optimizer=opt, loss={'policy_out':'categorical_crossentropy', 'value_out':'mse'})

        return model


    def store(self,state,action,next_state,traf,id_,route_keeper,term=0):
        reward = 0
        done = False

        lat,lon = next_state[-1][:2]

        dist = getClosestAC_Distance(self,[lat,lon,id_],traf,route_keeper)

        if dist < 10 and dist > 3:
            reward = -0.1 + 0.05*(dist/10)

        if dist < 3:
            reward = -1
            done = True

        d_goal = dist_goal(next_state[-1,:],traf,id_)


        if d_goal < 5 and done == False:
            reward = 0
            done = True


        if term == 1:

            reward= -1
            done = True

        if term == 2:
            reward = 0
            done = True


        state = self.normalize_that(state,'state').reshape(1,self.state_size*self.lag_values)
        next_state = self.normalize_that(next_state,'state').reshape(1,self.state_size*self.lag_values)



        try:
            self.experience[id_].append((state,action,reward,next_state,done))
        except:
            self.experience[id_] = [(state,action,reward,next_state,done)]


        if done:
            self.memory.append(self.experience[id_])

            del self.experience[id_]



    def train(self):

        """Grab samples from batch to train the network"""


        episodes = random.sample(self.memory,self.batch_size)

        for transitions in episodes:
            episode_length = len(transitions)
            state = np.array([rep[0] for rep in transitions]).reshape(len(transitions),self.state_size*self.lag_values)
            next_state  = np.array([rep[3] for rep in transitions]).reshape(len(transitions),self.state_size*self.lag_values)
            reward = np.array([rep[2] for rep in transitions])
            action = np.array([rep[1] for rep in transitions])
            done = np.array([rep[4] for rep in transitions])

            route = int(state[0][2]*2)
            discounted_rewards = self.discount(reward)

            policy,values = self.model.predict(state)

            advantages = np.zeros((episode_length, self.action_size))

            for i in range(episode_length):
                advantages[i][action[i]] = discounted_rewards[i] - values[i]

            self.model.fit({'states':state}, {'policy_out':advantages,'value_out':discounted_rewards}, epochs=1, verbose=0)


    def load(self, name):
        self.model.load_weights(name)


    def save(self,best=False):


        if best:

            self.model.save_weights('best_model.h5')


        else:

            self.model.save_weights('model.h5')


    # action implementation for the agent
    def act(self,state):
        rte = int(state[-1][3])
        state = self.normalize_that(state,'state')
        state = state.reshape((1,self.state_size))
        policy,value = self.model.predict(state)
        a = np.random.choice(self.action_size,1,p=policy.flatten())[0]

        return self.speeds[a],a

    def update(self,traf,index,route_keeper):
        """calulate reward and determine if terminal or not"""
        T = 0
        type_ = 0
        dist = getClosestAC_Distance(self,[traf.lat[index],traf.lon[index],traf.id[index]],traf,route_keeper)

        if dist < 3:
            T = True
            type_ = 1


        d_goal = dist_goal([traf.lat[index],traf.lon[index]],traf,traf.id[index])

        if d_goal < 5 and T == 0:
            T = True
            type_ = 2

        return T,type_
