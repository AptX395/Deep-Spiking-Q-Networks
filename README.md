# Deep-Spiking-Q-Networks

The source code of paper _Human-Level Control through Directly-Trained Deep Spiking Q-Networks_.

Besides the implementation of our Deep Spiking Q-Network (DSQN), we also reproduced the vanilla Deep Q-Network (DQN) proposed by [(Mnih et al. 2015)](https://doi.org/10.1038/nature14236) and conversion-based Spiking Neural Network (SNN) proposed by [(Tan,  Patel,  and  Kozma 2021)](https://ojs.aaai.org/index.php/AAAI/article/view/17180).

## Requirements

For DSQN and the vanilla DQN:

- Python 3.8.11
- PyTorch 1.8.2 LTS
- SpikingJelly
- Cupy
- Gym[atari,accept-rom-license]
- PyYAML
- Tensorboard

For conversion-based SNN:

- Python 3.8.11
- PyTorch 1.8.2 LTS
- TensorFlow-GPU 1.14
- OpenAI Baselines
- Gym[atari,accept-rom-license]
- PyYAML
- Matplotlib
- Pandas
- TensorboardX
- Sklearn

## Structure

This project could be devided into three parts：

1. The codes for DSQN and the vanilla DQN are under the **root** directory of this project.

    ```
    │  .gitignore
    │  LICENSE
    │  log_config.yaml
    │  README.md
    │  run.py	# Program entry.
    │
    ├─agent	# The module for RL agents.
    │      agent.py
    │      __init__.py
    │
    ├─environment	# The module for Gym Atari environments.
    │     atari_wrappers.py
    │     wrappers.py
    │     __init__.py
    │
    ├─neural_network	# The module for the vanilla DQN and our DSQN.
    │      neural_network.py
    │      __init__.py
    │
    └─replay_memory	# The module for experience replay memory.
          replay_memory.py
          __init__.py
    ```

2. The codes for the conversion-based SNN are under `conversion_based_snn` directory.

    ```
    │  convert.py	# Program entry.
    │  LICENSE
    │  README.md
    │
    └─bindsnet	# The module for implementing the conversion method.
        │  utils.py
        │  __init__.py
        │
        ├─analysis
        │      pipeline_analysis.py
        │      plotting.py
        │      visualization.py
        │      __init__.py
        │
        ├─conversion
        │      conversion.py
        │      __init__.py
        │
        ├─datasets
        │      alov300.py
        │      collate.py
        │      dataloader.py
        │      davis.py
        │      preprocess.py
        │      README.md
        │      spoken_mnist.py
        │      torchvision_wrapper.py
        │      __init__.py
        │
        ├─encoding
        │      encoders.py
        │      encodings.py
        │      loaders.py
        │      __init__.py
        │
        ├─environment
        │      environment.py
        │      __init__.py
        │
        ├─evaluation
        │      evaluation.py
        │      __init__.py
        │
        ├─learning
        │      learning.py
        │      reward.py
        │      __init__.py
        │
        ├─models
        │      models.py
        │      __init__.py
        │
        ├─network
        │      monitors.py
        │      network.py
        │      nodes.py
        │      topology.py
        │      __init__.py
        │
        ├─pipeline
        │      action.py
        │      base_pipeline.py
        │      dataloader_pipeline.py
        │      environment_pipeline.py
        │      __init__.py
        │
        └─preprocessing
                preprocessing.py
                __init__.py
    ```

3. The codes for ploting images are under `utils` directory.

    ```
        plot.py	# Program entry.
    ```

## Usage

For example, you could use following command to train our DSQN on Atari game Breakout with a single GPU by default:

```
python run.py --model=Dsqn --env_id=BreakoutNoFrameskip-v4
```

To train the vanilla DQN on Breakout, use:

```
python run.py --model=Dqn --env_id=BreakoutNoFrameskip-v4
```

To convert the vanilla DQN, which has been pre-trained on Breakout and saved under `model` directory by default, to SNN, first you need to locate under the `conversion_based_SNN` directory. Then, execute the following command:

```
python convert.py --game=BreakoutNoFrameskip-v4 --model=../model/Dqn_BreakoutNoFrameskip-v4_[time].pkl
```

More detailed arguments could be found in `run.py` and `conversion_based_snn/convert.py`. The default values of all arguments are as same as those reported in our paper.

## Acknowledgement

The codes under `conversion_based_snn` directory were forked from the open-source code of [(Tan,  Patel,  and  Kozma 2021)](https://ojs.aaai.org/index.php/AAAI/article/view/17180), which could be accessed at [here](https://github.com/WeihaoTan/bindsnet-1). We completed the reproduction based on their open-source code. Thanks a lot for their work!
