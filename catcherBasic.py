import numpy as np
import tflearn

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.conv import conv_2d, max_pool_2d, highway_conv_2d
from tflearn.layers.normalization import local_response_normalization, batch_normalization
from statistics import mean, median
from collections import Counter
from ple.games.catcher import Catcher
from ple import PLE

game = Catcher()
env = PLE(game, fps=30, display_screen=True)  # environment interface to game
env.init()
print(env.getActionSet()) #97 i 100
print(env.getScreenDims())

LR = 1e-3
#print(env.observation_space)
#print(env.action_space)
#env.reset()
goal_steps = 10000
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
            values = [player_x, player_vel, fruit_x]

            game_memory.append([values, action])
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
                training_data.append([data[0], output])
        progress = progress + 1
        print(progress)
        env.reset_game()
        scores.append(score)

    training_data_save = np.array(training_data)
    np.save('catcher.npy', training_data_save)

    print('Average accepted score:', mean(accepted_scores))
    print('Median score for accepted scores:', median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data


def neural_network_model(input_size):
    network = input_data(shape=[None, input_size, 1], name='input')

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
    X = np.array([i[0] for i in training_data]).reshape(-1, 3, 1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size = len(X[0]))

    model.fit({'input': X}, {'targets': y}, n_epoch=3, snapshot_step=500, show_metric=True,
              run_id='openai_learning', validation_set=0.1)
    return model



training_data = initial_population()
#training_data = np.load('catcher.npy')
model = train_model(training_data)
model.save('catcherBasic.model')
#model = neural_network_model(4)
#model.load('catcherBasic.model')
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
            player_vel = list(game_state_val)[1]
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
            action = model.predict(prev_obs.reshape(-1, 3, 1))[0]
            game_state = env.getGameState()
            game_state_val = game_state.values()
            player_x = list(game_state_val)[0]
            player_vel = list(game_state_val)[1]
            fruit_x = list(game_state_val)[2]
            fruit_y = list(game_state_val)[3]
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
        values = np.array([player_x, player_vel, fruit_x])
        prev_obs = values
        game_memory.append([values, new_action])
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