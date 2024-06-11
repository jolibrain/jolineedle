# Experiments Log

## Starting point (nov 2023):

- navigate over patches in the image
- can only move to adjacent patches
- supervised over generated trajectories with data augmentation
- Model = GPT + YOLOX, vision backbone shared between the two models
- Dataset = Lard

**Results** (on 100 images - Lard): **94.5%** patches found, **80.2%** MAP

**Results** (on full test set - Lard): **89.6%** patches found, **78.5%** MAP

## Supervised experiments

- Ablation study on positional encoding:
    - Removing positional encoding degrades model performances, the model needs more steps to find the runway once it's on the horizon.
    - Only positional encoding is not sufficient, the model goes at the correct place in the image but does not find the runway.

**Results** (on 100 images - Lard)
- _without pos enc_: **69.5%** patches found, **55%** MAP
- _with pos enc_: **86.7%** patches found, **73%** MAP

- Different backbones for GPT & YOLOX, GPT has a smaller backbone (yolox nano). With positional encoding this yields the best results so far. GPT benefits from its personal backbone trained only on decision.

**Results** (on 100 images - Lard): **95.7%** patches found, **84.2%** MAP

**Results** (on full test set - Lard): **92.6%** patches found, **84.6%** MAP

YOLOX-s MAP: **97.7%**

Hypothesis H1: Decision does not require a backbone as large as for detection. So limiting computing resources spent on detection would be the main lever to spare computing resources

## Reinforcement learning

- replace supervised learning by the "reinforce" algorithm.
- actions are learnt from model rollouts and rewards from the environment

- First training uses the same conditions as supervised training (2024/02/14)

**Results** (on 100 images - Lard): **96.5%** patches found

**Results** (on full test set - Lard): **91.9%** patches found

### Stop action

So far the model has no stop condition and continues the search for a fixed number of step. Here we add a STOP action, so that the model can decide wether to stop or continue the search. If the model stops after finding all the patches, it gets a big reward, otherwise it receives a penalty for each missed patch.

The STOP action could not be implemented easily with the supervised paradigm. With reinforcement learning, the model can use the STOP action only when appropriate, and avoiding it if it's a risk for performances.

**Results** (on 100 images - Lard): **91%** patches found

**Results** (on full test set - Lard): **90.2%** patches found

- Longer sequence (2024/02/19): now that we have stop action, we don't need to restrain the search length anymore, the model should just stop earlier. -> set `max_ep_len` to 20

2024/02/20: model stops immediately after first move, it seems like reward normalization doesn't work for `batch_size == 1` -> added param `reward_norm` to enable or disable reward normalization

**Results** (on full test set - Lard): **97.3%** patches found, **98.6%** bbox found, **16.7** mean episode length. The model stops by itself 99% of the time and misses a bbox 5% of the time by stopping.

- Simpler reward (2024/04): as the reward gets pretty complicated with the STOP action, we introduce a discount factor and change the reward to a simpler one: model gets a reward each time it finds a patch containing a bounding box.

**Results** (on full test set - Lard):

### Adding detection

We train a detection model (YOLOX) jointly with the decision model, just as in the supervised pipeline. The detection model is training on patches with bbox + randomly sampled negative patches.

TODO **Results**

## Training time

It seems like the training time is Disk bound (and maybe CPU-bound) rather than GPU bound. Further benchmarks are required to identify which part takes longer.

As of 2024-02-16, we're around 3sec/it, and 3 days for a full 100000 iters training at batch size 4


