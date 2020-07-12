import gym
import numpy as np
import cv2
import tflearn
import random

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.conv import conv_2d, max_pool_2d, highway_conv_2d
from tflearn.layers.normalization import local_response_normalization, batch_normalization
from statistics import mean, median
from collections import Counter
from ple.games.flappybird import FlappyBird
from ple.games.catcher import Catcher
from ple import PLE

game = FlappyBird()
env = PLE(game, fps=30, display_screen=True)  # environment interface to game
env.init()
print(env.getActionSet()) #97 i 100
#119 za flappy
print(env.getScreenDims())

LR = 1e-3
#env = gym.make('SpaceInvaders-v0')
#print(env.observation_space)
#print(env.action_space)
#env.reset()
goal_steps = 10000
score_requirement = 30
initial_games = 1000


def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    progress = 0
    for _ in range(initial_games):
        score = 0
        game_memory = []
        for _ in range(goal_steps):
            game_state = env.getGameState()
            game_state_val = game_state.values()
            player_y = list(game_state_val)[0]
            next_pipe_dist_to_player = list(game_state_val)[2]
            next_pipe_top_y = list(game_state_val)[3]
            next_pipe_bottom_y = list(game_state_val)[4]

            if next_pipe_bottom_y <= player_y:
                env.act(119)
                action = 119
            elif next_pipe_top_y >= player_y:
                env.act(None)
                action = None
            else:
                distance_top = player_y - next_pipe_top_y
                distance_bottom = next_pipe_bottom_y - player_y
                if distance_top > distance_bottom:
                    env.act(119)
                    action = 119
                elif distance_bottom > distance_top:
                    env.act(None)
                    action = None
                else:
                    randomNum = random.randint(0, 1)
                    if randomNum == 0:
                        env.act(119)
                        action = 119
                    else:
                        env.act(None)
                        action = None
            observation = cv2.cvtColor(cv2.resize(env.getScreenRGB(), (80, 80)), cv2.COLOR_BGR2GRAY)
            game_memory.append([observation, action])
            score = env.score()
            if env.game_over(): break

        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 119:
                    output = [1, 0]
                else:
                    output = [0, 1]
                training_data.append([data[0], output])
        progress = progress + 1
        print(progress)
        env.reset_game()
        scores.append(score)

    # just in case you wanted to reference later
    training_data_save = np.array(training_data)
    np.save('flappy2.npy', training_data_save)

    # some stats here, to further illustrate the neural network magic!
    print('Average accepted score:', mean(accepted_scores))
    print('Median score for accepted scores:', median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data


def neural_network_model(input_size, input_size2):
    network = input_data(shape=[None, input_size, input_size2, 1], name='input')

    network = conv_2d(network, 32, 3, activation='relu', strides=4)
    network = conv_2d(network, 64, 5, activation='relu', strides=2)
    network = conv_2d(network, 64, 3, activation='relu', strides=1)
    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.5)

    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy',
                         name='targets')
    model = tflearn.DNN(network)

    return model


def train_model(training_data, model=False):
    print(len(training_data[0][0]))
    print(len(training_data[0][0][0]))
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), len(training_data[0][0][0]), 1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size = len(X[0]), input_size2 = len(X[0][0]))

    model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True,
              run_id='openai_learning', validation_set=0.1)
    return model



#training_data = initial_population()
training_data = np.load('flappy2.npy')
model = train_model(training_data)
model.save('flappy2.model')
#model = neural_network_model(80, 80)
#model.load('flappy.model')
scores = []
accepted_scores = []
choices = []
for each_game in range(100):
    score = 0
    game_memory = []
    prev_obs = []
    for _ in range(goal_steps):
        if len(prev_obs) == 0:
            game_state = env.getGameState()
            game_state_val = game_state.values()
            player_y = list(game_state_val)[0]
            next_pipe_dist_to_player = list(game_state_val)[2]
            next_pipe_top_y = list(game_state_val)[3]
            next_pipe_bottom_y = list(game_state_val)[4]

            if next_pipe_bottom_y <= player_y:
                env.act(119)
                new_action = 119
            elif next_pipe_top_y >= player_y:
                env.act(None)
                new_action = None
            else:
                distance_top = player_y - next_pipe_top_y
                distance_bottom = next_pipe_bottom_y - player_y
                if distance_top > distance_bottom:
                    env.act(119)
                    new_action = 119
                elif distance_bottom > distance_top:
                    env.act(None)
                    new_action = None
                else:
                    randomNum = random.randint(0, 1)
                    if randomNum == 0:
                        env.act(119)
                        new_action = 119
                    else:
                        env.act(None)
                        new_action = None
        else:
            action = model.predict([prev_obs.reshape(80, 80, 1)])[0]
            new_action = np.argmax(action)
            if new_action == 0:
                new_action = 119
                env.act(119)
            else:
                new_action = None
                env.act(None)


        choices.append(new_action)
        new_observation = cv2.resize(env.getScreenGrayscale(), (80, 80))
        prev_obs = new_observation
        game_memory.append([new_observation, new_action])
        score = env.score()
        if env.game_over():
            break
    env.reset_game()
    scores.append(score)
    if score >= score_requirement:
        accepted_scores.append(score)

print('Average Score:', sum(scores) / len(scores))
print('Success rate:', len(accepted_scores) / len(scores))
print(score_requirement)
