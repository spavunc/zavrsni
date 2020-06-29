import gym
import numpy as np
import cv2
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from statistics import mean, median
from collections import Counter

LR = 1e-3
env = gym.make('Qbert-v0')
print(env.observation_space)
print(env.action_space)
env.reset()
goal_steps = 1000
score_requirement = 250
initial_games = 1000


def initial_population():
    # [OBS, MOVES]
    training_data = []
    # all scores:
    scores = []
    # just the scores that met our threshold:
    accepted_scores = []
    # iterate through however many games we want:
    progress = 0
    for _ in range(initial_games):
        score = 0
        # moves specifically from this environment:
        game_memory = []
        # previous observation that we saw
        prev_observation = []
        # for each frame in 200
        for _ in range(goal_steps):
            # choose random action (0 or 1)
            action = env.action_space.sample()
            # do it!
            observation, reward, done, info = env.step(action)
            inx, iny, inc = env.observation_space.shape

            observation = cv2.resize(cv2.cvtColor(observation, cv2.COLOR_RGBA2GRAY), (int(inx/8), int(iny/8)))

            # notice that the observation is returned FROM the action
            # so we'll store the previous observation here, pairing
            # the prev observation to the action we'll take.
            if len(prev_observation) > 0:
                # prev_observation = prev_observation.flatten()
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score += reward
            if done: break

        # IF our score is higher than our threshold, we'd like to save
        # every move we made
        # NOTE the reinforcement methodology here.
        # all we're doing is reinforcing the score, we're not trying
        # to influence the machine in any way as to HOW that score is
        # reached.
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                # convert to one-hot (this is the output layer for our neural network)
                if data[1] == 1:
                    output = [0, 1, 0, 0, 0, 0]
                elif data[1] == 0:
                    output = [1, 0, 0, 0, 0, 0]
                elif data[1] == 2:
                    output = [0, 0, 1, 0, 0, 0]
                elif data[1] == 3:
                    output = [0, 0, 0, 1, 0, 0]
                elif data[1] == 4:
                    output = [0, 0, 0, 0, 1, 0]
                elif data[1] == 5:
                    output = [0, 0, 0, 0, 0, 1]

                # saving our training data
                training_data.append([data[0], output])
        progress = progress + 1
        print(progress)
        # reset env to play again
        env.reset()
        # save overall scores
        scores.append(score)

    # just in case you wanted to reference later
    training_data_save = np.array(training_data)
    np.save('atari.npy', training_data_save)

    # some stats here, to further illustrate the neural network magic!
    print('Average accepted score:', mean(accepted_scores))
    print('Median score for accepted scores:', median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data


def neural_network_model(input_size, input_size2):
    network = input_data(shape=[None, input_size, input_size2, 1], name='input')

    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)


    network = fully_connected(network, 6, activation='softmax')
    network = regression(network, optimizer='momentum', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network)

    return model


def train_model(training_data, model=False):
    print(len(training_data[0]))
    print(len(training_data[0][0]))
    print(len(training_data[0][0][0]))
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), len(training_data[0][0][0]), 1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size = len(X[0]), input_size2 = len(X[0][0]))

    model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True,
              run_id='openai_learning')
    return model


training_data = initial_population()
model = train_model(training_data)
model.save('myModel.tflearn')
#model = neural_network_model(20 ,26)
#model.load('myModel.tflearn')
scores = []
choices = []

for each_game in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    for _ in range(goal_steps):
        env.render()
        if len(prev_obs) == 0:
            action = env.action_space.sample()
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1, 20, 26, 1))[0])

        choices.append(action)
        new_observation, reward, done, info = env.step(action)
        inx, iny, inc = env.observation_space.shape
        new_observation = cv2.resize(cv2.cvtColor(new_observation, cv2.COLOR_RGBA2GRAY), (int(inx / 8), int(iny / 8)))
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score += reward
        if done:
            break
    scores.append(score)

print('Average Score:', sum(scores) / len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1) / len(choices), choices.count(0) / len(choices)))
print('choice 2:{}  choice 3:{}'.format(choices.count(2) / len(choices), choices.count(3) / len(choices)))
print('choice 4:{}  choice 5:{}'.format(choices.count(4) / len(choices), choices.count(5) / len(choices)))
print(score_requirement)
