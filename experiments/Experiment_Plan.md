# Experimental Planning Document

This document outlines a detailed experiment pipeline to evaluate LLM/VLM-based policies in reinforcement learning tasks.

---

## Phase 1: Prompting Strategy Experiments

### Experiment 1.1: Structured vs. Unstructured Observations

- **Description:**  
  Evaluate whether LLM agents perform better with structured (e.g. JSON) observation data compared to raw, unstructured text.
  
  **Goal & Evaluation:**  
  - **Goal:** Identify which observation representation leads to higher cumulative rewards and more stable policy behavior.
  - **Evaluation:** Track average episode reward, episode length, and token efficiency (ratio of prompt size to performance gain).

- **Method/Experiment Design:**  
  - Test at least three input variations:
    1. Fully structured JSON representation.
    2. Semi-structured text (e.g. key-value pairs without full JSON syntax).
    3. Unstructured raw text.
  - Randomize trial order and control for environment randomness.
  - Compare performance metrics statistically over multiple runs (e.g. at least 30 episodes per variation).

- **Agents:**  
  - LLM Agent (text-only) with different prompt variants.

- **Environments:**  
  - MiniGrid-Empty-5x5-v0 (from the MiniGrid environments list)  
    *Justification:* A simple, controlled environment isolating the effect of prompt format.

---

### Experiment 1.2: Natural Language vs. Raw Data Representations

- **Description:**  
  Compare LLM performance when processing natural language descriptions of observations, rewards, and actions versus handling raw numerical data converted to text.
  
  **Goal & Evaluation:**  
  - **Goal:** Determine if natural language descriptions enhance decision-making and sample efficiency.
  - **Evaluation:** Measure average episode reward, sample efficiency, and conduct qualitative assessments of the generated reasoning.

- **Method/Experiment Design:**  
  - Create prompt variations where numerical observations are either directly converted into text or paraphrased into natural language descriptions.  
  - Run controlled experiments on the same underlying observations to isolate the effect of language conversion.
  - Include cross-validation with repeated trials for robustness (e.g. 20–30 episodes per variant).

- **Agents:**  
  - LLM Agent (text-only) using the two observation formatting strategies.

- **Environments:**  
  - MountainCar-v0 (from the Gymnasium environments list)  
    *Justification:* Its continuous control demands nuanced abstraction from raw data.

---

### Experiment 1.3: Effect of History Length on Decision Quality

- **Description:**  
  Investigate how varying observation history (e.g., full history, sliding window, or summarized context) affects the LLM’s planning and performance.
  
  **Goal & Evaluation:**  
  - **Goal:** Identify the optimal balance between providing sufficient context and avoiding irrelevant noise.
  - **Evaluation:** Compare cumulative rewards, episode lengths, and processing latency between the different history strategies.

- **Method/Experiment Design:**  
  - Implement several experimentation variations:
    1. Full history provided at every timestep.
    2. A sliding-window mechanism of fixed length.
      2.1 Order by recency
      2.2 Order by reward size
    3. Triggered summarization after a set number of steps or on exceeding a reward threshold.
  - Design controlled experiments with systematic change of one variable (history mode) at a time.
  - Use consistent evaluation across at least 30 episodes to ensure statistical significance.

- **Agents:**  
  - LLM Agent (text-only) with different context handling strategies.

- **Environments:**
  - MiniGrid-DoorKey-5x5-v0
    *Justification:* Requires multi-step planning where context plays a pivotal role.
  - BabyAI-GoToImpUnlock (use if DoorKey proves too easy)

---

## Phase 2: Comparative Evaluation with Traditional RL Baselines

### Experiment 2.1: LLM Agent vs. Conventional RL in Discrete Navigation

- **Description:**  
  Benchmark the LLM agent against established RL agents (e.g., PPO, DQN) on a controlled navigation task.
  
  **Goal & Evaluation:**  
  - **Goal:** Contrast emergent reasoning, sample efficiency, and final performance between LLM prompting strategies and gradient-based RL.
  - **Evaluation:** Measure task completion rate, cumulative rewards, and assess decision interpretability via textual rationales.

- **Method/Experiment Design:**  
  - Conduct multiple trials for each agent configuration (LLM prompt strategies vs. traditional RL parameters). (TODO: SELECT LLM AGENT VARIATIONS BASED ON EXPERIMENTS)
  - Analyze variance in performance under identical initial conditions.
  - Optionally record internal decision logs from the LLM for post hoc qualitative review.
  
- **Agents:**  
  - LLM Agent (text-only) with optimized prompt configurations.
  - Baseline Agents: PPO or DQN implemented through standard RL frameworks.

- **Environments:**  
  - BabyAI-GoToRedBall (from the BabyAI environments list)  
    *Justification:* This task provides a natural language grounding challenge ideal for comparing policy robustness.

---

### Experiment 2.2: Balance in Continuous Domains

- **Description:**  
  Compare LLM agents (via converted natural language prompts) with standard RL baselines on continuous control tasks.
  
  **Goal & Evaluation:**  
  - **Goal:** Determine whether converting continuous inputs to natural language retains sufficient fidelity for effective control.
  - **Evaluation:** Assess cumulative rewards, convergence times, and robustness across repeated runs.

- **Method/Experiment Design:**  
  - Develop baseline transformation protocols to convert sensor data into natural language.
  - Test different levels of granularity in the conversion (detailed vs. summarized descriptions).
  - Run side-by-side comparisons ensuring same random seed initialization for fair evaluation.
  
- **Agents:**  
  - LLM Agent (text-only with engineered prompt modifications).
  - Baseline Agents: DDPG or PPO.
  
- **Environments:**  
  - Reacher and CarRacing (from the Gymnasium environments list)  
    *Justification:* Both offer challenges in continuous control, testing the abstraction abilities of LLM-driven policies.

---

## Phase 3: VLM Experiments

### Experiment 3.1: Integration of Visual Observations

- **Description:**  
  Explore the capabilities of a multi-modal VLM Agent when provided with both visual inputs and textual descriptions.
  
  **Goal & Evaluation:**  
  - **Goal:** Assess improvements in navigation and control when visual information supplements text.
  - **Evaluation:** Compare cumulative rewards, time-to-goal metrics, and conduct qualitative analysis of action rationales.
  
- **Method/Experiment Design:**  
  - Develop an interface to feed concurrent visual and textual data to the VLM agent.
  - Experiment with different fusion strategies to balance modalities (e.g., early vs. late fusion).
  - Compare against a text-only LLM baseline under identical conditions using repeated episodes.
  
- **Agents:**  
  - VLM Agent (multi-modal).
  - Comparison: LLM Agent (text-only).
  
- **Environments:**  
  - CarRacing (from the Gymnasium environments list) or visually enhanced MiniGrid variants.
  - *Justification:* CarRacing inherently benefits from visual perception, providing a natural test bed for multi-modal integration.

---

## Phase 4: Ablation Studies

### Experiment 4.1: Prompt Component Ablation

- **Description:**  
  Systematically remove or vary key components in the prompt (e.g., reasoning instructions, context summaries, explicit goal definitions) to measure their contributions.
  
  **Goal & Evaluation:**  
  - **Goal:** Identify which prompt elements are critical for robust decision-making.
  - **Evaluation:** Analyze changes in cumulative rewards and task completion rates, and perform qualitative textual analysis of decision outputs.
  
- **Method/Experiment Design:**  
  - Develop a series of prompt templates with one or more components removed.
  - Conduct experiments in a controlled environment, running each variant sufficiently (minimum 30 episodes) to capture performance variance.
  - Compare against a full, optimized prompt baseline.
  
- **Agents:**  
  - LLM Agent (text-only) using modified prompt templates.
  
- **Environments:**  
  - MiniGrid-Empty-5x5-v0 and BabyAI-GoToDoor (from the BabyAI environments list)  
    *Justification:* This pairing allows isolation of language grounding effects in both simple and complex task scenarios.

---

## Additional Experiments (time permitting)

### System Prompt Refinement – Optimizing Prompts for Specific Tasks

- **Description:**  
  This experiment explores a dynamic system where the agent automatically identifies the task type and generates task-specific prompts. By tailoring prompts on the fly, the agent is expected to achieve improved performance across varied tasks.

  **Goal & Evaluation:**  
  - **Goal:** To determine if automated, task-oriented prompt refinements lead to higher agent performance as measured by rewards and task efficiency.  
  - **Evaluation:** Monitor task completion rates, cumulative rewards, convergence times, and qualitative analysis of the generated task-specific prompts.

- **Method/Experiment Design:**  
  - **Automated Task Identification:** Implement a module that analyzes the incoming observation data to categorize the task (e.g., navigation, object manipulation, planning).  
  - **Dynamic Prompt Generation:** Develop a mechanism to generate tailored prompt templates based on the identified task type.  
  - **Performance Thresholds & Reward-Based Refinement:** Design a feedback loop where the agent’s performance is monitored. If performance thresholds are not met, the system refines the prompt through an automated reward-based adjustment process.  
  - **Variations to Test:**  
    1. No prompt refinement (static prompt baseline).  
    2. Rule-based prompt generation (predefined templates for each task-type).  
    3. Adaptive prompt refinement that changes in real-time based on performance metrics.
  - **Controlled Trials:** Run multiple episodes (e.g., 30+ per variant) in environments that span multiple task types.

- **Agents:**  
  - LLM Agent equipped with the automated prompt refinement module.  
  - For comparisons, a baseline version of the LLM Agent with a fixed prompt is used.

- **Environments:**  
  - A mix of BabyAI and MiniGrid tasks that cover diverse domains (navigation, door unlocking, object interaction).  
    *Justification:* These environments emphasize distinct tasks which clearly benefit from customized prompting, allowing us to observe the impact of prompt optimization on agent behavior.

---

### Skill Generalization – Skill Vector Database Lookup

- **Description:**  
  This experiment investigates whether integrating a skill vector database can help the agent generalize across unseen tasks. By retrieving relevant skills based on past experience, the agent may exhibit enhanced performance in novel or varied scenarios.

  **Goal & Evaluation:**  
  - **Goal:** To quantify the effectiveness of a skill vector database in enabling cross-task generalization, thereby allowing the agent to leverage learned skill embeddings for improved performance on new tasks.  
  - **Evaluation:** Evaluate task success rate, time-to-goal, cumulative rewards, and sample efficiency when the agent is presented with unseen tasks versus tasks within the training distribution.

- **Method/Experiment Design:**  
  - **Database Implementation:**  
    - Construct a skill vector database that encodes key features of learned skills from earlier tasks (e.g., navigation, exploration, manipulation).  
    - Use similarity metrics to retrieve vectors that match the requirements of the current task.
  - **Integration with the Agent:**  
    - Modify the agent’s decision-making pipeline to incorporate the retrieved skill vectors as additional context for action selection.
  - **Controlled Experiments:**  
    - Compare performance on standard tasks (seen during training) against performance on a set of new, unseen tasks.
    - Test multiple variations for similarity thresholds in the lookup process.
  - **Feedback and Adaptation:**  
    - Introduce a reinforcement signal where successful retrievals reinforce the corresponding skill vectors, refining the database over time.

- **Agents:**  
  - LLM Agent with an integrated skill vector database lookup mechanism.  
  - Optionally, include a baseline without skill vector integration to compare generalization capabilities.

- **Environments:**  
  - Primarily BabyAI tasks with an emphasis on reasoning and object manipulation, where subtle differences between tasks can highlight generalization capability.  
    *Justification:* BabyAI tasks offer a range of skill requirements, making it an ideal setting to test whether retrieved skills across similar contexts can boost performance on new tasks.

---

## Additional Note

- **Proxy Experiments:**  
  - We may want to begin with simple environments like CartPole or FrozenLake (from the Gymnasium environments) to validate basic hypotheses before scaling up complexity.

- **Metrics:**  
  - In addition to traditional reward-based metrics, evaluate interpretability (clarity of internal reasoning), token efficiency (performance relative to prompt size), and behavior diversity (variance in actions over time)
