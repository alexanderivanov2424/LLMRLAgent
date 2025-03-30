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

# TODO


class ExperimentData:

  def __init__(self, exp_name):
    self.exp_name = exp_name

    self.meta_data = {}

    # TODO all other experiment data can be initialized here

  def get_file_path(self):
    # TODO we want more information to be stored in the experiment data object at construction to make it easier to disambiguate
    return os.path.join(SAVE_DIR, self.exp_name + ".json")

  def save(self):
    data = {}
    data["exp_name"] = self.exp_name
    data["meta_data"] = self.meta_data


    path = self.get_file_path()
    with open(path, 'w', encoding='utf-8') as fp:
      json.dump(data, fp, ensure_ascii=False, indent=4, sort_keys=True)
    

  def load(EXP_NAME):
    experiment = ExperimentData(EXP_NAME)

    path = experiment.get_file_path()
    with open(path, 'r', encoding='utf-8') as fp:
      data = json.load(fp)

    # redundant, here we would normally populate the rest of the data
    experiment.exp_name = data["exp_name"]
    experiment.meta_data = data["meta_data"]
    return experiment
  
  # TODO more helper functions to save new bits of data into the experiment blob

  def log_meta_data(self, key, value):
    self.meta_data[key] = value