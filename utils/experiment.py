import os
import json

"""
Class for loading/saving experimental data. 


We want this to have some basic support for storing:
- meta information about an experiment (name, agents, env, etc...)
- agent actions, rewards, episodes, wallclock time, etc
- any other agent related data (specific prompts, hyperparameters, etc...)

This class should also be able to load experiments so that we can more easily plot data outside the cluster

"""

SAVE_DIR = "./experiment_data"


KEY_EXP_NAME = "exp_name"
KEY_META_DATA = "meta_data"
KEY_AGENT = "agents"
KEY_AGENT_EPISODE = "episodes"
KEY_AGENT_EPISODE_REWARDS = "rewards"


class ExperimentData:

  def __init__(self, exp_name):
    self.data = {}

    self.exp_name = exp_name
    self.data[KEY_EXP_NAME] = exp_name
    self.data[KEY_META_DATA] = {}

    self.data[KEY_AGENT] = {}

  def get_file_path(self):
    # TODO we want more information to be stored in the experiment data object at construction to make it easier to disambiguate
    return os.path.join(SAVE_DIR, self.exp_name + ".json")

  def save(self):
    path = self.get_file_path()
    with open(path, 'w', encoding='utf-8') as fp:
      json.dump(self.data, fp, ensure_ascii=False, sort_keys=True)
    

  def load(experiment_name):
    experiment = ExperimentData(experiment_name)

    path = experiment.get_file_path()
    with open(path, 'r', encoding='utf-8') as fp:
      data = json.load(fp)

    # redundant, here we would normally populate the rest of the data
    experiment.exp_name = data[KEY_EXP_NAME]
    experiment.data = data
    return experiment
  

  ###############################
  
  # TODO more helper functions to save new bits of data into the experiment blob

  def log_meta_data(self, key, value):
    self.data[KEY_META_DATA][key] = value

  

  def log_agent_episode_rewards(self, agent, episode_number, rewards_list):
    agent_name = agent.get_agent_name()

    # ensure we have a dictionary for all the agent data
    if not agent_name in self.data[KEY_AGENT]:
      self.data[KEY_AGENT][agent_name] = {}

    # ensure we have a key for episode data in the agent dictionary
    if not KEY_AGENT_EPISODE in self.data[KEY_AGENT][agent_name]:
      self.data[KEY_AGENT][agent_name][KEY_AGENT_EPISODE] = {}

    # ensure that we have an entry for this episode  
    if not episode_number in self.data[KEY_AGENT][agent_name][KEY_AGENT_EPISODE]:
      self.data[KEY_AGENT][agent_name][KEY_AGENT_EPISODE][episode_number] = {}
    
    self.data[KEY_AGENT][agent_name][KEY_AGENT_EPISODE][episode_number][KEY_AGENT_EPISODE_REWARDS] = rewards_list
