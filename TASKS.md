# Tasks

## Domain Prep for LLMRL Agent

- [ ] Make sure all the following domains are compatible with the LLMRL Agent
  - [ ] MiniGrid Doorkey
  - [ ] MiniGrid Emptyenv
  - [ ] MiniGrid Lavagapenv

  - [ ] CartPole
  - [ ] Lunar Lander
  - [ ] Reacher

## Experimental Logging

Currently experiments log cumulative reward and total reward per episode.

- [ ] Identify if we need anything else based on papers in related works
- [ ] Log wall clock time average per episode (maybe per step)

## LLMRL Agent Implementation

- [ ] Feed trajectory history back into prompt. Agent has memory of full trajectory so far.
  - [ ] Produce data on handful of simple domains
    - [ ] MiniGrid Doorkey
    - [ ] MiniGrid Emptyenv
    - [ ] MiniGrid Lavagapenv
  - [ ] Identify alternative prompts, create new config for each
  - [ ] Test and produce data for all alternative prompts on handful of simple domains
    - [ ] MiniGrid Doorkey
    - [ ] MiniGrid Emptyenv
    - [ ] MiniGrid Lavagapenv

- [ ] Memory LLMRL, create new Agent with explicit memory block
  - [ ] implement logic for updating agent memory as text (reference related work for strategies)
  - [ ] Produce data on handful of simple domains
    - [ ] MiniGrid Doorkey
    - [ ] MiniGrid Emptyenv
    - [ ] MiniGrid Lavagapenv
  - [ ] Identify alternative prompts, create new config for each
  - [ ] Test and produce data for all alternative promps on handful of simple domains
    - [ ] MiniGrid Doorkey
    - [ ] MiniGrid Emptyenv
    - [ ] MiniGrid Lavagapenv

- [ ] General collection of LLM Agent tests as described in `Experiment_Planning`
  - This should happen after at least one of the blocks above is finished
  - [ ] TODO break those out as tasks here


### Discrete Domains

- Generate baseline performance for handful of simple domains with several learning algorithms

- [ ] Random Agent
  - [ ] MiniGrid Doorkey
  - [ ] MiniGrid Emptyenv
  - [ ] MiniGrid Lavagapenv
  - [ ] MiniGrid BabyAI GoToImpUnlock
- [ ] LLMRL Agent
  - [ ] MiniGrid Doorkey
  - [ ] MiniGrid Emptyenv
  - [ ] MiniGrid Lavagapenv
  - [ ] MiniGrid BabyAI GoToImpUnlock
- [ ] Q-Learning Agent
  - [ ] MiniGrid Doorkey
  - [ ] MiniGrid Emptyenv
  - [ ] MiniGrid Lavagapenv
  - [ ] MiniGrid BabyAI GoToImpUnlock
- [ ] General DQN
  - [ ] Choose general DQN hyperparameters
    - [ ] MiniGrid Doorkey
    - [ ] MiniGrid Emptyenv
    - [ ] MiniGrid Lavagapenv
    - [ ] MiniGrid BabyAI GoToImpUnlock
- [ ] Specialized DQN
  - [ ] Optimize hyper parameters per environment
    - [ ] MiniGrid Doorkey
    - [ ] MiniGrid Emptyenv
    - [ ] MiniGrid Lavagapenv
    - [ ] MiniGrid BabyAI GoToImpUnlock
  - [ ] Produce training data with optimized hyper parameters
    - [ ] MiniGrid Doorkey
    - [ ] MiniGrid Emptyenv
    - [ ] MiniGrid Lavagapenv
    - [ ] MiniGrid BabyAI GoToImpUnlock

### Continuous Domains

- [ ] Random Agent
  - [ ] CartPole
  - [ ] Lunar Lander
  - [ ] Reacher
- [ ] LLMRL Agent
  - [ ] CartPole
  - [ ] Lunar Lander
  - [ ] Reacher
- [ ] General PPO
  - [ ] Choose general DQN hyperparameters
    - [ ] CartPole
    - [ ] Lunar Lander
    - [ ] Reacher
- [ ] Specialized PPO
  - [ ] Optimize hyper parameters per environment
    - [ ] CartPole
    - [ ] Lunar Lander
    - [ ] Reacher
  - [ ] Produce training data with optimized hyper parameters
    - [ ] CartPole
    - [ ] Lunar Lander
    - [ ] Reacher


## Multitask Discrete Experiments

- [ ] Train data for LLMRL Agent across combination of discrete domains
  - [ ] MiniGrid Doorkey
  - [ ] MiniGrid Emptyenv
  - [ ] MiniGrid Lavagapenv
  - [ ] MiniGrid BabyAI GoToImpUnlock
- [ ] Train data for DQN Agent across combination of discrete domains
  - [ ] MiniGrid Doorkey
  - [ ] MiniGrid Emptyenv
  - [ ] MiniGrid Lavagapenv
  - [ ] MiniGrid BabyAI GoToImpUnlock


## Multitask Continuous Experiments

- [ ] Train data for LLMRL Agent across domains
  - [ ] CartPole
  - [ ] Lunar Lander
  - [ ] Reacher
- [ ] Train data for PPO Agent across domains
  - [ ] CartPole
  - [ ] Lunar Lander
  - [ ] Reacher


## More Agents To Consider

  More agents we can run through all the environments if we have time:

- [ ] TD3

## More Domains To Consider

  More domains we can test against if we can setup the corresponding wrapper for the LLM
  
- Car Racing
- Taxi
- Frozen Lake

## Notes

- We may want to save the random seeds used in each experiment for fair evaluation
  - Yes, we should use the same seed everywhere for reproducability. For now envs use a seed of 0
- George said PPO is generally better in continuous domains, but was this referring to the observation or action space?? Should we run both DQN and PPO in both sets of experiments for robustness?

