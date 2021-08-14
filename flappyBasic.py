import numpy as np
import cv2
import tflearn
import random

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter
from ple.games.flappybird import FlappyBird
from ple import PLE

game = FlappyBird()
env = PLE(game, fps=30, display_screen=True)
env.init()
print(env.getActionSet())
#119 za flappy
print(env.getScreenDims())

LR = 1e-3
#print(env.observation_space)
#print(env.action_space)
#env.reset()
goal_steps = 100000
score_requirement = 30
initial_games = 500


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
            print(game_state.values())
            game_state_val = game_state.values()
            player_y = list(game_state_val)[0]
            player_vel = list(game_state_val)[1]
            next_pipe_dist_to_player = list(game_state_val)[2]
            next_pipe_top_y = list(game_state_val)[3]
            next_pipe_bottom_y = list(game_state_val)[4]
            next_next_pipe_dist_to_player = list(game_state_val)[5]
            next_next_pipe_top_y = list(game_state_val)[6]
            next_next_pipe_bottom_y = list(game_state_val)[7]

            if next_pipe_dist_to_player <= 10:
                distance_top = player_y - next_pipe_top_y
                distance_bottom = next_pipe_bottom_y - player_y
                distance_top_next = player_y - next_next_pipe_top_y
                distance_bottom_next = next_next_pipe_bottom_y - player_y
                if distance_top_next > distance_bottom_next and distance_top > 15:
                    env.act(119)
                    action = 119
                elif distance_bottom_next > distance_top_next and distance_bottom > 15:
                    env.act(None)
                    action = None
                else:
                    if distance_top > distance_bottom:
                        env.act(119)
                        action = 119
                    elif distance_bottom > distance_top:
                        env.act(None)
                        action = None
            elif next_pipe_bottom_y <= player_y:
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
            if env.game_over():
                break
            observation = cv2.cvtColor(cv2.resize(env.getScreenRGB(), (80, 80)), cv2.COLOR_BGR2GRAY)
            values = [player_y, player_vel, next_pipe_dist_to_player, next_pipe_bottom_y, next_pipe_top_y,
                      next_next_pipe_dist_to_player, next_next_pipe_top_y, next_next_pipe_bottom_y]
            if len(prev_observation) > 0:
                game_memory.append([values, action, observation])
            prev_observation = observation
            score = env.score()

        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 119:
                    output = [1, 0]
                else:
                    output = [0, 1]
                training_data.append([data[0], output, data[2]])
        progress = progress + 1
        print(progress)
        env.reset_game()
        scores.append(score)

    training_data_save = np.array(training_data)
    np.save('flappy6.npy', training_data_save)

    print('Average accepted score:', mean(accepted_scores))
    print('Median score for accepted scores:', median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data


def neural_network_model(input_size):
    network = input_data(shape=[None, input_size,  1], name='input')

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


    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy',
                         name='targets')
    model = tflearn.DNN(network)

    return model


def train_model(training_data, model=False):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
    #X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), len(training_data[0][0][0]), 1)
    Y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size = len(X[0]))

    model.fit(X, Y, n_epoch=3, snapshot_step=500, show_metric=True,
              run_id='openai_learning', validation_set=0.1)

    return model



#training_data = initial_population()

#training_data = np.load('flappy5.npy')
#trained = neural_network_model(80, 80)
#trained.load('flappy3.model')
#model = train_model(training_data)
#model.save('flappyBasic.model')


model = neural_network_model(8)
model.load('flappyBasic.model')
scores = []
accepted_scores = []
choices = []
for each_game in range(100):
    score = 0
    game_memory = []
    prev_obs = []
    for _ in range(goal_steps):
        game_state = env.getGameState()
        game_state_val = game_state.values()
        player_y = list(game_state_val)[0]
        player_vel = list(game_state_val)[1]
        next_pipe_dist_to_player = list(game_state_val)[2]
        next_pipe_top_y = list(game_state_val)[3]
        next_pipe_bottom_y = list(game_state_val)[4]
        next_next_pipe_dist_to_player = list(game_state_val)[5]
        next_next_pipe_top_y = list(game_state_val)[6]
        next_next_pipe_bottom_y = list(game_state_val)[7]
        if len(prev_obs) == 0:
            randomNum = random.randint(0, 1)
            if randomNum == 0:
                env.act(119)
                new_action = 119
            else:
                env.act(None)
                new_action = None
        else:
            action = model.predict(prev_obs.reshape(-1, 8, 1))[0]
            new_action = np.argmax(action)
            if new_action == 0:
                new_action = 119
                env.act(119)
            else:
                new_action = None
                env.act(None)


        choices.append(new_action)
        new_observation = cv2.cvtColor(cv2.resize(env.getScreenRGB(), (80, 80)), cv2.COLOR_BGR2GRAY)
        values = np.array([player_y, player_vel, next_pipe_dist_to_player, next_pipe_bottom_y, next_pipe_top_y,
                      next_next_pipe_dist_to_player, next_next_pipe_top_y, next_next_pipe_bottom_y])
        game_memory.append([values, new_action, new_observation])
        prev_obs = values
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