# Conversion-based SNN

The codes under the current directory were forked from the open-source code of [(Tan,  Patel,  and  Kozma 2021)](https://ojs.aaai.org/index.php/AAAI/article/view/17180), which could be accessed at [here](https://github.com/WeihaoTan/bindsnet-1).

## Requirements

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

## Usage

First, you need a pre-trained vanilla DQN. For example, you could train a vanilla DQN on Atari game Breakout by execute the following command under the **root** directory of this project:

```
python run.py --model=Dqn --env_id=BreakoutNoFrameskip-v4
```

The trained vanilla DQN would be saved under `model` directory.

Next, you could convert the pre-trained vanilla DQN to SNN under `conversion_based_snn` directory as follows:

```
python convert.py --game=BreakoutNoFrameskip-v4 --model=../model/Dqn_BreakoutNoFrameskip-v4_[time].pkl
```

