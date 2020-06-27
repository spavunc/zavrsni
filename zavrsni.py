import gym
import random
import numpy as np
from PIL.Image import core as _imaging
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.conv import conv_2d, max_pool_2d
from statistics import mean, median
from collections import Counter

LR = 2.5e-3
env = gym.make('Qbert-v0')
print(env.observation_space)
print(env.action_space)
env.reset()
goal_steps = 500
score_requirement = 200
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


def neural_network_model():
    convnet = input_data(shape=[None, 210, 160, 3], name='input')

    convnet = conv_2d(convnet, 32, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = fully_connected(convnet, 128, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 256, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 512, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 256, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 128, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 6, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(convnet)

    return model


def train_model(training_data, model=False):
    X = np.array([i[0] for i in training_data]).reshape(-1, 210, 160, 3)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model()

    model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True,
              run_id='openai_learning')
    return model


training_data = initial_population()
model = train_model(training_data)

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
            action = np.argmax(model.predict(prev_obs.reshape(-1, 210, 160, 3))[0])

        choices.append(action)
        new_observation, reward, done, info = env.step(action)
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
