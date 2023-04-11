# coding = utf-8

"""
Forked from the open-source code of paper Strategy and Benchmark for Converting Deep Q-Networks to Event-Driven Spiking Neural
Networks.
"""

from baselines import deepq
from baselines import bench
from baselines import logger
from baselines.common.atari_wrappers import make_atari
from baselines.common.tf_util import save_state
from baselines.common.tf_util import get_session

import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf

import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from time import time as t

from bindsnet.conversion import Permute, ann_to_snn
from bindsnet.network.topology import MaxPool2dConnection
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes, plot_voltages

import random
import logging
from logging.config import dictConfig
from datetime import datetime
import yaml
from bindsnet.network import nodes

CUR_FILE_PATH = os.path.split(os.path.realpath(__file__))[0]
LOG_DIR = os.path.join(CUR_FILE_PATH, "log")
LOG_CONFIG_FILE = os.path.join(CUR_FILE_PATH, "..", "log_config.yaml")
RESULT_DIR = os.path.join(CUR_FILE_PATH, "result")

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

if not os.path.exists(RESULT_DIR):
    os.mkdir(RESULT_DIR)

parser = argparse.ArgumentParser()
parser.add_argument('--game', type=str, default='video_pinball')
parser.add_argument('--model', type=str, default='pytorch_video_pinball.pt')
# parser.add_argument('--episode', type=int)
# parser.add_argument('--time', type=int)
# parser.add_argument('--seed', type=int)
# parser.add_argument('--percentile', type=float)

parser.add_argument("--episode", type=int, default=30)
parser.add_argument("--time", type=int, default=500)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--percentile", type=float, default=99.9)
parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--node_type", type=str, default="SubtractiveResetIFNodes", choices=["SubtractiveResetIFNodes",
                                                                                         "LIFNodes"])
parser.add_argument("--tau", type=float, default=2.0)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def set_logger():
    with open(LOG_CONFIG_FILE, mode="r") as file:
        log_config = yaml.load(file, yaml.SafeLoader)

    # Set the file name of the log file.
    log_name = f"{args.game}_{time_str}"
    log_config["handlers"]["file_handler"]["filename"] = os.path.join(LOG_DIR, f"{log_name}.log")
    dictConfig(log_config)


def set_random_seed(seed, is_using_cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tf.set_random_seed(seed)

    if is_using_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def policy(q_values, eps):
    A = np.ones(len(q_values), dtype=float) * eps / len(q_values)
    best_action = torch.argmax(q_values)
    A[best_action] += (1.0 - eps)
    return A, best_action


def convert(seed=10086, time=500, percentile=99.9, game='video_pinball', model='pytorch_video_pinball.pt', episode=50,
            node_type="SubtractiveResetIFNodes", tau=2.0):
    file_logger = logging.getLogger("file_logger")

    # seed = seed
    n_examples = 15000
    time = time
    epsilon = 0.05
    percentile = percentile
    episode = episode

    # if torch.cuda.is_available():
    #     torch.set_default_tensor_type('torch.cuda.FloatTensor')
    #     device = 'cuda'
    # else:
    #     device = 'cpu'

    device = torch.device(args.device)

    # print("device", device)
    # print("game", game, "episode", episode, "time", time, "seed", seed, "percentile", percentile)

    name = ''.join([g.capitalize() for g in game.split('_')])
    env = make_atari(game, max_episode_steps=18000)  
    env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
    env = deepq.wrap_atari_dqn(env)
    env.seed(seed) 
    n_actions = env.action_space.n

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            # 1 input image channel, 6 output channels, 5x5 square convolution
            # kernel

            # self.conv1 = nn.Conv2d(4, 32, 8, stride=4, padding=2)
            self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
            self.relu1 = nn.ReLU()
            # self.pad2 = nn.ConstantPad2d((1, 2, 1, 2), value=0)
            # self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=0)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
            self.relu2 = nn.ReLU()
            # self.pad3 = nn.ConstantPad2d((1, 1, 1, 1), value=0)
            # self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=0)
            self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
            self.relu3 = nn.ReLU()
            #self.perm = Permute((0, 2, 3, 1))
            # self.perm = Permute((1, 2, 0))
            # self.fc1 = nn.Linear(7744, 512)
            self.flatten1 = nn.Flatten()
            self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=512)
            self.relu4 = nn.ReLU()
            # self.fc2 = nn.Linear(512, n_actions)
            self.fc2 = nn.Linear(in_features=512, out_features=n_actions)

        def forward(self, x):
            x = x / 255.0
            x = self.relu1(self.conv1(x))
            # x = self.pad2(x)
            x = self.relu2(self.conv2(x))
            # x = self.pad3(x)
            x = self.relu3(self.conv3(x))
            # x = x.permute(0, 2, 3, 1).contiguous()
            # x = x.view(-1, self.num_flat_features(x))
            x = self.flatten1(x)
            x = self.relu4(self.fc1(x))
            x = self.fc2(x)
            return x

        def show(self, x):
            x = x
            x = self.relu1(self.conv1(x))
            # x = self.pad2(x)
            x = self.relu2(self.conv2(x))
            # x = self.pad3(x)
            x = self.relu3(self.conv3(x))
            # x = x.permute(0, 2, 3, 1).contiguous()
            # x = x.view(-1, self.num_flat_features(x))
            x = self.flatten1(x)
            x = self.relu4(self.fc1(x))
            x = self.fc2(x)
            return torch.max(x, 1)[1].data[0]

        def num_flat_features(self, x):
            size = x.size()[1:]  # all dimensions except the batch dimension
            num_features = 1
            for s in size:
                num_features *= s
            return num_features

    class Dqn(nn.Module):
        def __init__(self):
            super(Dqn, self).__init__()

            # Define the neural network layers.
            self._conv1 = nn.Sequential(nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
                                        nn.ReLU())
            self._conv2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
                                        nn.ReLU())
            self._conv3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
                                        nn.ReLU())
            self._fc1 = nn.Sequential(nn.Flatten(),
                                      nn.Linear(in_features=64 * 7 * 7, out_features=512),
                                      nn.ReLU())
            self._fc2 = nn.Linear(in_features=512, out_features=n_actions)

        def forward(self, x):
            x = self._conv1(x)
            x = self._conv2(x)
            x = self._conv3(x)
            x = self._fc1(x)
            x = self._fc2(x)
            return x

    model_path = model
    dqn = Dqn()
    dqn.load_state_dict(torch.load(model_path))
    ANN_model = Net()
    # ANN_model.load_state_dict(torch.load(model_path))

    # Load state-dicts.
    ANN_model.conv1.load_state_dict(dqn._conv1[0].state_dict())
    ANN_model.relu1.load_state_dict(dqn._conv1[1].state_dict())
    ANN_model.conv2.load_state_dict(dqn._conv2[0].state_dict())
    ANN_model.relu2.load_state_dict(dqn._conv2[1].state_dict())
    ANN_model.conv3.load_state_dict(dqn._conv3[0].state_dict())
    ANN_model.relu3.load_state_dict(dqn._conv3[1].state_dict())
    ANN_model.flatten1.load_state_dict(dqn._fc1[0].state_dict())
    ANN_model.fc1.load_state_dict(dqn._fc1[1].state_dict())
    ANN_model.relu4.load_state_dict(dqn._fc1[2].state_dict())
    ANN_model.fc2.load_state_dict(dqn._fc2.state_dict())

    ANN_model.eval()
    ANN_model = ANN_model.to(device)

    images = []
    cnt = 0

    for epi in range(1000):
        cur_images = []
        cur_cnt = 0
        obs, done = env.reset(), False                                       
        episode_rew = 0
        while not done:
            #env.render()
            image = torch.from_numpy(obs[None]).permute(0, 3, 1, 2)
            cur_images.append(image.detach().numpy())
            actions_value = ANN_model(image.to(device)).cpu()[0]            
            probs, best_action = policy(actions_value, epsilon)
            action = np.random.choice(np.arange(len(probs)), p=probs)

            obs, rew, done, info = env.step(action)
            cur_cnt += 1
        if info['ale.lives'] == 0:
            if cur_cnt + cnt < n_examples:
                cnt += cur_cnt
                images += cur_images
            else:
                # print("normalization image cnt", cnt)

                if len(images) < n_examples:
                    images += cur_images

                file_logger.info(f"normalization image cnt: {cnt}")

                break

    images = torch.from_numpy(np.array(images)).reshape(-1, 4, 84, 84).float() / 255

    
    # SNN = ann_to_snn(ANN_model, input_shape=(4, 84, 84), data=images.to(device), percentile = percentile)

    if node_type == "SubtractiveResetIFNodes":
        SNN = ann_to_snn(ANN_model, input_shape=(4, 84, 84), data=images.to(device), percentile=percentile)
    elif node_type == "LIFNodes":
        SNN = ann_to_snn(ANN_model, input_shape=(4, 84, 84), data=images.to(device), percentile=percentile,
                         node_type=nodes.LIFNodes, rest=0.0, tc_decay=tau)

    SNN = SNN.to(device)

    for l in SNN.layers:
        if l != 'Input':
            SNN.add_monitor(
                Monitor(SNN.layers[l], state_vars=['s', 'v'], time=time), name=l
            )
    
    for c in SNN.connections:
        if isinstance(SNN.connections[c], MaxPool2dConnection):
            SNN.add_monitor(
                Monitor(SNN.connections[c], state_vars=['firing_rates'], time=time), name=f'{c[0]}_{c[1]}_rates'
            )

    # f = open("game" + game + "episode" + str(episode) + "time" + str(time) + "percentile" + str(percentile) + ".csv",'a')
    f = open(os.path.join(RESULT_DIR, f"game_{game}_episode_{episode}_time_{time}_percentile_{percentile}.csv"), "a")
    game_cnt = 0
    mix_cnt = 0
    spike_cnt = 0
    cnt = 0
    rewards = np.zeros(episode)

    episode_avg_firing_rates = list()
    total_action_num = 0
    total_spike_num = 0

    while(game_cnt < episode):

        obs, done = env.reset(), False

        avg_firing_rates = list()

        while not done:
            image = torch.from_numpy(obs[None]).permute(0, 3, 1, 2).float()  / 255 
            image = image.to(device)

            ANN_action = ANN_model.show(image.to(device))

            inpts = {'Input': image.repeat(time, 1, 1, 1, 1)}
            SNN.run(inputs=inpts, time=time)

            spikes = {
                l: SNN.monitors[l].get('s') for l in SNN.monitors if 's' in SNN.monitors[l].state_vars
            }
            voltages = {
                l: SNN.monitors[l].get('v') for l in SNN.monitors if 'v' in SNN.monitors[l].state_vars
            }

            # actions_value = spikes['12'].sum(0).cpu() + voltages['12'][time -1].cpu()
            actions_value = spikes["10"].sum(0).cpu() + voltages["10"][time - 1].cpu()
            action = torch.max(actions_value, 1)[1].data.numpy()[0]

            # spike_actions_value = spikes['12'].sum(0).cpu()
            spike_actions_value = spikes["10"].sum(0).cpu()
            spike_action = torch.max(spike_actions_value, 1)[1].data.numpy()[0]


            cnt += 1
            if ANN_action == action:
                mix_cnt += 1
            if ANN_action == spike_action:
                spike_cnt += 1

            probs, best_action = policy(actions_value[0], epsilon)
            action = np.random.choice(np.arange(len(probs)), p=probs)

            if action == np.argmax(probs):
                total_spike = 0
                total_count = 0

                for spike_data in spikes.values():
                    total_spike += torch.sum(spike_data).item()
                    total_count += spike_data.numel()

                avg_firing_rate = total_spike / total_count
                avg_firing_rates.append(avg_firing_rate)
                total_action_num += 1
                total_spike_num += total_spike

            SNN.reset_state_variables()
            obs, rew, done, info = env.step(action)

        if info['ale.lives'] == 0:
            rewards[game_cnt] = info['episode']['r']
            # print("Episode " +str(game_cnt) +" reward", rewards[game_cnt])
            # print("cnt", cnt, "mix", mix_cnt / cnt, "spike", spike_cnt / cnt)

            episode_avg_firing_rate = np.mean(avg_firing_rates)
            episode_avg_firing_rates.append(episode_avg_firing_rate)
            file_logger.info(f"episode: {game_cnt}\treward: {rewards[game_cnt]}")
            file_logger.info(f"cnt: {cnt}\tmix: {mix_cnt / cnt}\tspike: {spike_cnt / cnt}")

            f.write(str(rewards[game_cnt]) + ", " + str(mix_cnt / cnt) + ", " + str(spike_cnt / cnt) + "\n")
            game_cnt += 1
            mix_cnt = 0
            spike_cnt = 0
            cnt = 0
        elif 'TimeLimit.truncated' in info:
            if info['TimeLimit.truncated'] == True:
                rewards[game_cnt] = info['episode']['r']
                # print("Episode " +str(game_cnt) +" reward", rewards[game_cnt])
                # print("cnt", cnt, "mix", mix_cnt / cnt, "spike", spike_cnt / cnt)

                episode_avg_firing_rate = np.mean(avg_firing_rates)
                episode_avg_firing_rates.append(episode_avg_firing_rate)
                file_logger.info(f"episode: {game_cnt}\treward: {rewards[game_cnt]}")
                file_logger.info(f"cnt: {cnt}\tmix: {mix_cnt / cnt}\tspike: {spike_cnt / cnt}")

                f.write(str(rewards[game_cnt]) + ", " + str(mix_cnt / cnt) + ", " + str(spike_cnt / cnt) + "\n")
                game_cnt += 1
                mix_cnt = 0
                spike_cnt = 0
                cnt = 0

    average_firing_rate = np.mean(episode_avg_firing_rates)
    spike_num_per_action = total_spike_num / total_action_num
    file_logger.info(f"average_firing_rate: {average_firing_rate}\tspike_num_per_action: {spike_num_per_action}")

    env.close()
    f.close()
    # print("Avg: ", np.mean(rewards))
    # output_str = "Avg: " + str(np.mean(rewards))

    file_logger.info(f"average_score: {np.mean(rewards)}\tstd: {np.std(rewards)}")


def main():
    set_logger()
    is_using_cuda = True if args.device == "cuda" else False
    set_random_seed(args.seed, is_using_cuda)
    convert(args.seed, args.time, args.percentile, args.game, args.model, args.episode, args.node_type, args.tau)


if __name__ == "__main__":
    main()
