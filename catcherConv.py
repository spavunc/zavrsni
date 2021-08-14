import numpy as np
import cv2
import tflearn

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.conv import conv_2d, max_pool_2d, highway_conv_2d
from tflearn.layers.normalization import local_response_normalization, batch_normalization
from statistics import mean, median
from collections import Counter
from ple.games.flappybird import FlappyBird
from ple.games.catcher import Catcher
from ple import PLE

game = Catcher()
env = PLE(game, fps=30, display_screen=True)
env.init()
print(env.getActionSet()) #97 i 100
print(env.getScreenDims())

LR = 1e-3
#env = gym.make('SpaceInvaders-v0')
#print(env.observation_space)
#print(env.action_space)
#env.reset()
goal_steps = 100000
score_requirement = 295
initial_games = 100


def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    progress = 0
    for _ in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            game_state = env.getGameState()
            game_state_val = game_state.values()
            player_x = list(game_state_val)[0]
            player_vel = list(game_state_val)[1]
            fruit_x = list(game_state_val)[2]
            fruit_y = list(game_state_val)[3]

            if fruit_x < player_x:
                env.act(97)
                action = 97
            elif fruit_x > player_x:
                env.act(100)
                action = 100
            else:
                env.act(None)
                action = None
            observation = cv2.cvtColor(cv2.resize(env.getScreenRGB(), (32, 32)), cv2.COLOR_BGR2GRAY)
            values = [player_x, player_vel, fruit_x, fruit_y]
            if len(prev_observation) > 0:
                game_memory.append([values, action, observation])
            prev_observation = observation
            score = env.score()
            if env.game_over(): break

        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 97:
                    output = [1, 0, 0]
                elif data[1] == 100:
                    output = [0, 1, 0]
                else:
                    output = [0, 0, 1]
                training_data.append([data[0], output, data[2]])
        progress = progress + 1
        print(progress)
        env.reset_game()
        scores.append(score)

    training_data_save = np.array(training_data)
    np.save('catcherTest.npy', training_data_save)

    print('Average accepted score:', mean(accepted_scores))
    print('Median score for accepted scores:', median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data


def neural_network_model(input_size, input_size2):
    network = input_data(shape=[None, input_size, input_size2, 1], name='input')

    network = conv_2d(network, 32, 3, activation='relu', strides=4)
    network = conv_2d(network, 64, 5, activation='relu', strides=2)
    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.5)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.5)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.5)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.5)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.5)

    network = fully_connected(network, 3, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy',
                         name='targets')
    model = tflearn.DNN(network)

    return model


def train_model(training_data, model=False):
    X = np.array([i[2] for i in training_data]).reshape(-1, 32, 32, 1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size = len(X[0]), input_size2 = len(X[0][0]))

    model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True,
              run_id='openai_learning', validation_set=0.1)
    return model



#training_data = initial_population()
#training_data = np.load('catcherTest.npy')
#model = train_model(training_data)
#model.save('catcherConv.model')
model = neural_network_model(32, 32)
model.load('catcherConv.model')
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
            player_x = list(game_state_val)[0]
            fruit_x = list(game_state_val)[2]
            fruit_y = list(game_state_val)[3]

            if fruit_x < player_x:
                env.act(97)
                new_action = 97
            elif fruit_x > player_x:
                env.act(100)
                new_action = 100
            else:
                env.act(None)
                new_action = None
        else:
            action = model.predict([prev_obs.reshape(32, 32, 1)])[0]
            new_action = np.argmax(action)
            if new_action == 0:
                new_action = 97
                env.act(97)
            elif new_action == 1:
                new_action = 100
                env.act(100)
            else:
                new_action = None
                env.act(None)


        choices.append(new_action)
        new_observation = cv2.cvtColor(cv2.resize(env.getScreenRGB(), (32, 32)), cv2.COLOR_BGR2GRAY)
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
