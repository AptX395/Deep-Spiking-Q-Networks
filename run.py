# coding = utf-8

import os
import argparse
import random
from logging.config import dictConfig
from datetime import datetime
import yaml
import numpy
import torch
from torch.utils.tensorboard import SummaryWriter
from gym.wrappers import Monitor
from spikingjelly.clock_driven.monitor import Monitor as SpikeMonitor
from environment import make_atari, wrap_deepmind
from neural_network import Dqn, Dsqn, DsqnSpike, DsqnIF
from replay_memory import ReplayMemory
from agent import DqnAgent, DsqnAgent

CUR_FILE_PATH = os.path.split(os.path.realpath(__file__))[0]
LOG_DIR = os.path.join(CUR_FILE_PATH, "log")
LOG_CONFIG_FILE = os.path.join(CUR_FILE_PATH, "log_config.yaml")
MODEL_DIR = os.path.join(CUR_FILE_PATH, "model")
MONITOR_DIR = os.path.join(CUR_FILE_PATH, "monitor")

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

if not os.path.exists(MONITOR_DIR):
    os.mkdir(MONITOR_DIR)

parser = argparse.ArgumentParser()

# Settings.
parser.add_argument("--model", type=str, default="Dsqn", choices=["Dqn", "Dsqn", "DsqnSpike", "DsqnIF"])
parser.add_argument("--env_id", type=str, default="BreakoutNoFrameskip-v4", choices=["AtlantisNoFrameskip-v4",
                                                                                     "BeamRiderNoFrameskip-v4",
                                                                                     "BoxingNoFrameskip-v4",
                                                                                     "BreakoutNoFrameskip-v4",
                                                                                     "CrazyClimberNoFrameskip-v4",
                                                                                     "GopherNoFrameskip-v4",
                                                                                     "JamesbondNoFrameskip-v4",
                                                                                     "KangarooNoFrameskip-v4",
                                                                                     "KrullNoFrameskip-v4",
                                                                                     "NameThisGameNoFrameskip-v4",
                                                                                     "PongNoFrameskip-v4",
                                                                                     "RoadRunnerNoFrameskip-v4",
                                                                                     "SpaceInvadersNoFrameskip-v4",
                                                                                     "StarGunnerNoFrameskip-v4",
                                                                                     "TennisNoFrameskip-v4",
                                                                                     "TutankhamNoFrameskip-v4",
                                                                                     "VideoPinballNoFrameskip-v4"])
parser.add_argument("--method", type=str, default="learn", choices=["learn", "play"])
parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--model_path", type=str)
parser.add_argument("--use_video_recorder", type=bool, default=False)
parser.add_argument("--use_spike_monitor", type=bool, default=False)

# Common hyperparameters.
parser.add_argument("--timestep_num", type=int, default=int(50e6))
parser.add_argument("--eval_freq", type=int, default=int(25e4))
parser.add_argument("--eval_episode_num", type=int, default=30)
parser.add_argument("--eval_epsilon", type=float, default=0.05)
parser.add_argument("--max_episode_step", type=int, default=int(18e3))
parser.add_argument("--minibatch_size", type=int, default=32)
parser.add_argument("--replay_memory_size", type=int, default=int(1e6))
parser.add_argument("--history_len", type=int, default=4)
parser.add_argument("--target_net_update_freq", type=int, default=int(1e4))
parser.add_argument("--discount_factor", type=float, default=0.99)
parser.add_argument("--action_repeat", type=int, default=4)
parser.add_argument("--update_freq", type=int, default=4)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--init_epsilon", type=float, default=1.0)
parser.add_argument("--final_epsilon", type=float, default=0.1)
parser.add_argument("--final_epsilon_frame", type=int, default=int(1e6))
parser.add_argument("--replay_start_size", type=int, default=int(5e4))

# The hyperparameters of `Dsqn`.
parser.add_argument("--T", type=int, default=64)
parser.add_argument("--tau", type=float, default=2.0)
parser.add_argument("--v_threshold", type=float, default=1.0)
parser.add_argument("--v_reset", type=float, default=0.0)
parser.add_argument("--surrogate_func", type=str, default="ATan", choices=["ATan", "Sigmoid"])
parser.add_argument("--alpha", type=float, default=2.0)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

if args.v_reset == -1.0:
    args.v_reset = None


def set_logger():
    with open(LOG_CONFIG_FILE, mode="r") as file:
        log_config = yaml.load(file, yaml.SafeLoader)

    # Set the file name of the log file.
    log_name = f"{args.model}_{args.env_id}_{args.method}_{time_str}"
    log_config["handlers"]["file_handler"]["filename"] = os.path.join(LOG_DIR, f"{log_name}.log")
    dictConfig(log_config)

    writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, log_name))
    return writer


def set_random_seed(seed, is_using_cuda):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    if is_using_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def learn(writer):
    # Create the Atari environments for learning and evaluation.
    train_env = make_atari(args.env_id, args.max_episode_step)
    train_env = wrap_deepmind(train_env, frame_stack=True)
    train_env.seed(args.seed)
    eval_env = make_atari(args.env_id, args.max_episode_step)
    eval_env = wrap_deepmind(eval_env, episode_life=False, clip_rewards=False, frame_stack=True)
    eval_env.seed(args.seed)

    replay_memory = ReplayMemory(args.replay_memory_size)  # Create the experience replay memory for learning.
    action_num = train_env.action_space.n  # As the output layer size of the neural networks.
    device = torch.device(args.device)

    # Create the neural networks and the reinforcement learning agent according to the parameter setting.
    if args.model == "Dqn":
        main_net = Dqn(args.history_len, action_num, args.learning_rate, device)
        target_net = Dqn(args.history_len, action_num, args.learning_rate, device)
        agent = DqnAgent(eval_env, main_net, writer, train_env, target_net, replay_memory)
    elif args.model == "Dsqn":
        main_net = Dsqn(args.history_len, action_num, args.learning_rate, device, args.T, args.tau, args.v_threshold,
                        args.v_reset, args.surrogate_func, args.alpha)
        target_net = Dsqn(args.history_len, action_num, args.learning_rate, device, args.T, args.tau, args.v_threshold,
                          args.v_reset, args.surrogate_func, args.alpha)
        agent = DsqnAgent(eval_env, main_net, writer, train_env, target_net, replay_memory)
    elif args.model == "DsqnSpike":
        main_net = DsqnSpike(args.history_len, action_num, args.learning_rate, device, args.T, args.tau, args.v_threshold,
                             args.v_reset, args.surrogate_func, args.alpha)
        target_net = DsqnSpike(args.history_len, action_num, args.learning_rate, device, args.T, args.tau, args.v_threshold,
                               args.v_reset, args.surrogate_func, args.alpha)
        agent = DsqnAgent(eval_env, main_net, writer, train_env, target_net, replay_memory)
    elif args.model == "DsqnIF":
        main_net = DsqnIF(args.history_len, action_num, args.learning_rate, device, args.T, args.v_threshold, args.v_reset,
                          args.surrogate_func, args.alpha)
        target_net = DsqnIF(args.history_len, action_num, args.learning_rate, device, args.T, args.v_threshold, args.v_reset,
                            args.surrogate_func, args.alpha)
        agent = DsqnAgent(eval_env, main_net, writer, train_env, target_net, replay_memory)

    agent.learn(args.timestep_num, args.replay_start_size, args.minibatch_size, args.discount_factor, args.update_freq,
                args.target_net_update_freq, args.eval_freq, args.eval_episode_num, args.eval_epsilon, args.init_epsilon,
                args.final_epsilon, args.final_epsilon_frame,
                model_path=os.path.join(MODEL_DIR, f"{args.model}_{args.env_id}_{time_str}.pkl"))
    train_env.close()
    eval_env.close()


def play(writer):
    # Create the Atari environment for evaluation.
    eval_env = make_atari(args.env_id, args.max_episode_step)
    eval_env = wrap_deepmind(eval_env, episode_life=False, clip_rewards=False, frame_stack=True)

    # Set up the monitor for video recording.
    if args.use_video_recorder:
        monitor_dir = os.path.join(MONITOR_DIR, f"{args.model}_{args.env_id}_{time_str}")

        if not os.path.exists(monitor_dir):
            os.mkdir(monitor_dir)

        eval_env = Monitor(eval_env, monitor_dir, video_callable=lambda episode_id: True, force=True)

    eval_env.seed(args.seed)
    action_num = eval_env.action_space.n  # As the output layer size of the neural network.
    device = torch.device(args.device)

    # Create the neural network and the reinforcement learning agent according to the parameter setting.
    if args.model == "Dqn":
        main_net = Dqn(args.history_len, action_num, args.learning_rate, device)
        agent = DqnAgent(eval_env, main_net, writer)
    elif args.model == "Dsqn":
        main_net = Dsqn(args.history_len, action_num, args.learning_rate, device, args.T, args.tau, args.v_threshold,
                        args.v_reset, args.surrogate_func, args.alpha)

        if args.use_spike_monitor:
            spike_monitor = SpikeMonitor(main_net, device=device, backend="torch")
            agent = DsqnAgent(eval_env, main_net, writer, spike_monitor=spike_monitor)
        else:
            agent = DsqnAgent(eval_env, main_net, writer)

    elif args.model == "DsqnSpike":
        main_net = DsqnSpike(args.history_len, action_num, args.learning_rate, device, args.T, args.tau, args.v_threshold,
                             args.v_reset, args.surrogate_func, args.alpha)
        agent = DsqnAgent(eval_env, main_net, writer)
    elif args.model == "DsqnIF":
        main_net = DsqnIF(args.history_len, action_num, args.learning_rate, device, args.T, args.v_threshold, args.v_reset,
                          args.surrogate_func, args.alpha)
        agent = DsqnAgent(eval_env, main_net, writer)

    agent.play(args.model_path, args.eval_episode_num, args.eval_epsilon)
    eval_env.close()


def main():
    writer = set_logger()
    is_using_cuda = True if args.device == "cuda" else False
    set_random_seed(args.seed, is_using_cuda)

    if args.method == "learn":
        learn(writer)
    elif args.method == "play":
        play(writer)

    writer.close()


if __name__ == "__main__":
    main()
