# Tasks

## Phase 2

(skipped phase 1 due to limited time)

### Experiment 2.1

- [ ] Choose general DQN hyperparameters

- [ ] Finalize Evaluation Metrics

- [ ] Finalize LLM Agent Action and Observation Space Prompts

- [ ] Evaluate LLM Agent
  - [ ] Test LLM Agent on Environments
    - [ ] MiniGrid Doorkey
    - [ ] MiniGrid Emptyenv
    - [ ] MiniGrid Lavagapenv
    - [ ] MiniGrid BabyAI GoToImpUnlock

- [ ] Evaluate General DQNs
  - [ ] Train General DQN on Environments
    - [ ] MiniGrid Doorkey
    - [ ] MiniGrid Emptyenv
    - [ ] MiniGrid Lavagapenv
    - [ ] MiniGrid BabyAI GoToImpUnlock

  - [ ] Test General DQN on Environments
    - [ ] MiniGrid Doorkey
    - [ ] MiniGrid Emptyenv
    - [ ] MiniGrid Lavagapenv
    - [ ] MiniGrid BabyAI GoToImpUnlock

- [ ] Evaluate Specialized DQNs
  - [ ] Train General DQN on Environments
    - [ ] MiniGrid Doorkey
    - [ ] MiniGrid Emptyenv
    - [ ] MiniGrid Lavagapenv
    - [ ] MiniGrid BabyAI GoToImpUnlock

  - [ ] Test General DQN on Environments
    - [ ] MiniGrid Doorkey
    - [ ] MiniGrid Emptyenv
    - [ ] MiniGrid Lavagapenv
    - [ ] MiniGrid BabyAI GoToImpUnlock

### Experiment 2.2

- [ ] Choose general PPO hyperparameters

- [ ] Finalize Evaluation Metrics

- [ ] Finalize LLM Agent Action and Observation Space Prompts

- [ ] Implement Gymnasium Reacher Wrapper

- [ ] Evaluate LLM Agent
  - [ ] Test LLM Agent on Environments
    - [ ] CartPole
    - [ ] Lunar Lander
    - [ ] Reacher

- [ ] Evaluate General PPOs
  - [ ] Train General PPO on Environments
    - [ ] CartPole
    - [ ] Lunar Lander
    - [ ] Reacher

  - [ ] Test General PPO on Environments
    - [ ] CartPole
    - [ ] Lunar Lander
    - [ ] Reacher

- [ ] Evaluate Specialized PPOs
  - [ ] Train General PPO on Environments
    - [ ] CartPole
    - [ ] Lunar Lander
    - [ ] Reacher

  - [ ] Test General PPO on Environments
    - [ ] CartPole
    - [ ] Lunar Lander
    - [ ] Reacher


# Current Experiment Plan

## Phase 2: Comparative Evaluation with Traditional RL Baselines

### Experiment 2.1: LLM Agent vs. Conventional RL in Discrete Domain

- **Description:**  
  Benchmark the LLM agent against established RL agents on a controlled navigation task.
  
  **Goal & Evaluation:**  
  - **Goal:** Contrast emergent reasoning, sample efficiency, and final performance between LLM prompting strategies and gradient-based RL.
  - **Evaluation:** Measure task completion rate, cumulative rewards, and decision interpretability (via reasoning traces).

- **Method/Experiment Design:**  
  - Run LLM Agent on each of the environments
  - Run "general" (ie. same hyperparameter for all envs) DQN on environments
  - Run "specialized" (ie tuned hyperparameter) DQN on environments

- **Environments:**
  - Gymnasium
    - NOTES
      - I excluded these (Taxi and Frozen Lake) in the tasks above due to their similarity to the MiniGrid Environments
    - Taxi
    - Frozen Lake
  
  - MiniGrid
    - Doorkey
    - Emptyenv
    - Lavagapenv
    - BabyAI/GoToImpUnlock
      - May require training curriculum or other more advanced techniques

---

### Experiment 2.2: Balance in Continuous Domains

- **Description:**  
  Compare LLM agents (via converted natural language prompts) with standard RL baselines on continuous control tasks.
  
  **Goal & Evaluation:**  
  - **Goal:** Determine whether converting continuous inputs to natural language retains sufficient fidelity for effective control.
  - **Evaluation:** Assess cumulative rewards, convergence times, and robustness across repeated runs.

- **Method/Experiment Design:**  
  - Run LLM Agent on each of the environments
    - We may need to experiment with truncated context window
  - Run "general" (ie. same hyperparameter for all envs) PPO on environments
  - Run "specialized" (ie tuned hyperparameter) PPO on environments

  - NOTES
    - We may want to save the random seeds used in each experiment for fair evaluation
    - George said PPO is generally better in continuous domains, but was this referring to the observation or action space?? Should we run both DQN and PPO in both sets of experiments for robustness?
  
- **Environments:**  
  - Implemented Wrappers:
    - CartPole (continuous observation, discrete action)
    - Lunar Lander (continuous observation, discrete action)
  - Unimplemented Wrappers:
    - Reacher (continuous observation, continuous action)
