import numpy as np
import gym
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.optimizers import Adam, SGD
import random




#parameters
batchSize = 25
memory_size = 50
learningrate=0.01
gamma = 0.1
epsilon = 0.1
maxStepsPerEpi = 1000
max_epi = 2500



#set gym env to lunalander
name = 'LunarLander-v2'
env = gym.make(name)
# Use random seed to get repeatable results.
env.seed(0)
numberAct = env.action_space.n

print("number of act is " + str(numberAct))

# set up Keras model for a 4 layer NN
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(40))
model.add(Activation('relu'))
model.add(Dense(40))
model.add(Activation('relu'))
model.add(Dense(numberAct))
model.add(Activation('linear'))
model.compile(loss='mean_squared_error',  optimizer=Adam(lr=learningrate))
print(model.summary())
print(env.observation_space.shape)


def actionChoosing(state, epsilon):
   
   # Use Greedy-epsilon exploration. Chooses action with the highest Q value with epsilon to choose a random action.
   
    r = np.random.uniform()  # from np.random
    if r < epsilon:
        return env.action_space.sample() # take random action
    else:
        #get optimal action by using argmax of q value vector at input state 
        stateQValue = predictQValue(state)
        action = np.argmax(stateQValue)
        return action


def predictQValue(state):
    inputState = np.empty([1, 1, 8])
    inputState[:] = state
    return model.predict(inputState)[0]

    
   
def getTargets(state, action, reward, nextState):
    #Returns target Q-values 
    
    currentStateQ = predictQValue(state)
    nextStateQ = predictQValue(nextState)

    targets = np.empty([1, numberAct])

    for i in range(numberAct):
        if i == action:
            targets[0][i] = reward + (gamma * np.max(nextStateQ))
        else:
            targets[0][i] = currentStateQ[i]
    return targets




class Memory(object):
    def __init__(self, memory_size=10000, experience_size=1):
        self.experiences = np.empty([0, experience_size], dtype=object)
        self.max_memory_size = memory_size

    def addExperience(self, experience):
        self.experiences = np.insert(self.experiences, 0,experience, axis=0)
        if len(self.experiences) > self.max_memory_size:
            self.experiences = np.delete(self.experiences,self.max_memory_size, axis=0)

    def sample_experiences(self, mini_batch_size):
        if(mini_batch_size > len(self.experiences)):
            toReplace = True
        else:
            toReplace = False
        sampleExp = self.experiences[np.random.choice(self.experiences.shape[0],mini_batch_size, replace=toReplace)]
        return sampleExp


def pack_experience(state, action, reward, nextState):
    experience = np.empty([0])
    experience = np.append(experience, state)
    experience = np.append(experience, [action])
    experience = np.append(experience, [reward])
    experience = np.append(experience, nextState)
    return experience


def unpack_experience(experience):
    state = experience[0:8]
    action = experience[8]
    reward = experience[9]
    nextState = experience[10:18]
    return state, action, reward, nextState


def learn_from_replay_memories(memory, sizeOfBatch):
    
    sampleBatch = memory.sample_experiences(sizeOfBatch)
    for exp in sampleBatch:
        state, action, reward, nextState = unpack_experience(exp)
        targets = getTargets(state, action, reward, nextState)
        x = np.empty([1, 1, 8])
        x[0][0] = state
        model.train_on_batch(x, targets)



memory = Memory(memory_size, 18)
finalReward = np.zeros(max_epi)


for epi in range(max_epi):
    state = env.reset()
    stepsTaken = 0
    epiDone = False
    while stepsTaken < maxStepsPerEpi and not epiDone:
        # Using the greedy-epsilon policy
        action = actionChoosing(state, epsilon)
        nextState, reward, epiDone, info = env.step(action)
        finalReward[epi] = finalReward[epi] + reward


        #write to memory
        experience = pack_experience(state, action, reward, nextState)
        memory.addExperience(experience)
        stepsTaken = stepsTaken + 1
        state = nextState
        
        # Learn from past experiences
        learn_from_replay_memories(memory, batchSize)

    print ("Episode %i reward = %0.2f" % (epi, finalReward[epi]))
    if not epi % 100 and epi:
        print ("------------------------------------------")
        print ("Last 100 episode avg = %0.2f" % np.average(
            finalReward[epi-100:epi]))
        print ("------------------------------------------")

    if epi == max_epi-1 or np.average(finalReward[epi-100:epi+1]) > 200:
        with open(r'allEpis.txt', 'w') as f:
            f.write(" ".join(map(str, finalReward)))
        with open(r'best100.txt', 'w') as f:
            f.write(" ".join(map(str, finalReward[epi-100:epi+1])))
        break

print ("Max episode reward = %0.2f" % np.max(finalReward))



