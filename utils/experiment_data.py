import os
import json
from json.decoder import JSONDecodeError
from typing import List
from agents.base_agent import BaseAgent

"""
Class for loading/saving experimental data. 


We want this to have some basic support for storing:
- meta information about an experiment (name, agents, env, etc...)
- agent actions, rewards, episodes, wallclock time, etc
- any other agent related data (specific prompts, hyperparameters, etc...)

This class should also be able to load experiments so that we can more easily plot data outside the cluster

"""

EXPERIMENT_SAVE_DIR = "./experiment_data"


KEY_EXP_NAME = "exp_name"
KEY_META_DATA = "meta_data"
KEY_AGENT = "agents"
KEY_AGENT_EPISODE = "episodes"
KEY_AGENT_EPISODE_REWARDS = "rewards"
KEY_AGENT_EPISODE_LENGTH = "ep_len"
KEY_AGENT_EPISODE_AVG_REWARD = "avg_reward"
KEY_AGENT_EPISODE_SUM_REWARD = "sum_reward"
KEY_AGENT_EPISODE_TIME_POLICY = "policy_time"
KEY_AGENT_EPISODE_TIME_UPDATE = "update_time"

# New key for storing multi-env evaluation results
KEY_AGENT_MULTI_ENV_EVAL_REWARDS = "multi_env_eval_rewards"


class ExperimentData:

    def __init__(self, exp_name):
        self.data = {}

        self.exp_name = exp_name
        self.data[KEY_EXP_NAME] = exp_name
        self.data[KEY_META_DATA] = {}

        self.data[KEY_AGENT] = {}

    def get_file_path(self):
        # TODO we want more information to be stored in the experiment data object at construction to make it easier to disambiguate
        return os.path.join(EXPERIMENT_SAVE_DIR, self.exp_name + ".json")

    def save(self):
        path = self.get_file_path()

        # make sure the directory exists
        os.makedirs(EXPERIMENT_SAVE_DIR, exist_ok=True)

        with open(path, "w", encoding="utf-8") as fp:
            json.dump(self.data, fp, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def load(experiment_name):
        experiment = ExperimentData(experiment_name)

        path = experiment.get_file_path()

        if not os.path.isfile(path):
            return experiment

        try:
            with open(path, "r", encoding="utf-8") as fp:
                data = json.load(fp)

            # redundant, here we would normally populate the rest of the data
            experiment.exp_name = data[KEY_EXP_NAME]
            experiment.data = data
        except JSONDecodeError as e:
            print("[Error] Failed to load experiment, check file for corruption")
            exit()
        return experiment


    def _get_agent_episode_dict(self, agent_ID: str, episode_number: int):
        # ensure we have a dictionary for all the agent data
        if not agent_ID in self.data[KEY_AGENT]:
            self.data[KEY_AGENT][agent_ID] = {}

        # ensure we have a key for episode data in the agent dictionary
        if not KEY_AGENT_EPISODE in self.data[KEY_AGENT][agent_ID]:
            self.data[KEY_AGENT][agent_ID][KEY_AGENT_EPISODE] = []

        # ensure that we have an entry for this episode
        episode_count = len(self.data[KEY_AGENT][agent_ID][KEY_AGENT_EPISODE])
        if episode_count < episode_number:
            print(
                "[WARNING] Logged episode number {episode_number} doesn't correspond to saved episode count {episode_count}. Did you skip an episode?"
            )

        # ensure that we have a dictionary for episode data
        if episode_count == episode_number:
            self.data[KEY_AGENT][agent_ID][KEY_AGENT_EPISODE].append({})

        return self.data[KEY_AGENT][agent_ID][KEY_AGENT_EPISODE][episode_number]

    ###############################
    # functions to save bits of data into the blob

    # The goal here is to not have to manually track the dictionary keys outside of this file

    def log_meta_data(self, key, value):
        self.data[KEY_META_DATA][key] = value

    def log_agent_episode_rewards(
        self, agent: BaseAgent, episode_number: int, rewards_list: List[float]
    ):
        agent_ID = agent.get_agent_ID()
        agent_episode = self._get_agent_episode_dict(agent_ID, episode_number)
        agent_episode[KEY_AGENT_EPISODE_REWARDS] = rewards_list

    def log_agent_episode_length(
        self, agent: BaseAgent, episode_number: int, length: int
    ):
        agent_ID = agent.get_agent_ID()
        agent_episode = self._get_agent_episode_dict(agent_ID, episode_number)
        agent_episode[KEY_AGENT_EPISODE_LENGTH] = length

    def log_agent_episode_reward_meta_stats(
        self, agent: BaseAgent, episode_number: int, reward_sum: float, reward_avg: float
    ):
        agent_ID = agent.get_agent_ID()
        agent_episode = self._get_agent_episode_dict(agent_ID, episode_number)
        agent_episode[KEY_AGENT_EPISODE_SUM_REWARD] = reward_sum
        agent_episode[KEY_AGENT_EPISODE_AVG_REWARD] = reward_avg

    def log_agent_episode_policy_time(
        self, agent: BaseAgent, episode_number: int, time: float
    ):
        agent_ID = agent.get_agent_ID()
        agent_episode = self._get_agent_episode_dict(agent_ID, episode_number)
        agent_episode[KEY_AGENT_EPISODE_TIME_POLICY] = time
    
    def log_agent_episode_update_time(
        self, agent: BaseAgent, episode_number: int, time: float
    ):
        agent_ID = agent.get_agent_ID()
        agent_episode = self._get_agent_episode_dict(agent_ID, episode_number)
        agent_episode[KEY_AGENT_EPISODE_TIME_UPDATE] = time

    def log_agent_multi_env_eval_rewards(
        self, agent_ID: str, eval_event_num: int, env_rewards_dict: dict
    ):
        """Logs rewards from multiple environments for a single evaluation event."""
        agent_episode = self._get_agent_episode_dict(agent_ID, eval_event_num)
        agent_episode[KEY_AGENT_MULTI_ENV_EVAL_REWARDS] = env_rewards_dict

    ###############################
    # functions to get out the data more easily (for plots, visuals)

    def get_agents(self):
        return self.data[KEY_AGENT].keys()

    def get_agent_epsiode_count(self, agent_ID):
        if not agent_ID in self.data[KEY_AGENT]:
            return 0
        if not KEY_AGENT_EPISODE in self.data[KEY_AGENT][agent_ID]:
            return 0
        return len(self.data[KEY_AGENT][agent_ID][KEY_AGENT_EPISODE])

    def get_agent_episode_rewards(self, agent_ID, episode_number):
        agent_episode = self._get_agent_episode_dict(agent_ID, episode_number)
        if KEY_AGENT_EPISODE_REWARDS in agent_episode:
            return agent_episode[KEY_AGENT_EPISODE_REWARDS]
        print("[Warning] No Episode Reward Data")
        return -1
    
    def get_agent_episode_length(self, agent_ID, episode_number):
        agent_episode = self._get_agent_episode_dict(agent_ID, episode_number)
        if KEY_AGENT_EPISODE_LENGTH in agent_episode:
            return agent_episode[KEY_AGENT_EPISODE_LENGTH]
        print("[Warning] No Episode Length Data")
        return -1
    
    def get_agent_episode_sum_reward(self, agent_ID, episode_number):
        agent_episode = self._get_agent_episode_dict(agent_ID, episode_number)
        if KEY_AGENT_EPISODE_SUM_REWARD in agent_episode:
            return agent_episode[KEY_AGENT_EPISODE_SUM_REWARD]
        print("[Warning] No Reward Sum Data")
        return -1

    def get_agent_episode_average_reward(self, agent_ID, episode_number):
        agent_episode = self._get_agent_episode_dict(agent_ID, episode_number)
        if KEY_AGENT_EPISODE_AVG_REWARD in agent_episode:
            return agent_episode[KEY_AGENT_EPISODE_AVG_REWARD]
        print("[Warning] No Reward Avg Data")
        return -1
    
    def get_agent_episode_policy_time(self, agent_ID, episode_number):
        agent_episode = self._get_agent_episode_dict(agent_ID, episode_number)
        if KEY_AGENT_EPISODE_LENGTH in agent_episode:
            return agent_episode[KEY_AGENT_EPISODE_TIME_POLICY]
        print("[Warning] No Episode Policy Time Data")
        return -1
    
    def get_agent_episode_update_time(self, agent_ID, episode_number):
        agent_episode = self._get_agent_episode_dict(agent_ID, episode_number)
        if KEY_AGENT_EPISODE_LENGTH in agent_episode:
            return agent_episode[KEY_AGENT_EPISODE_TIME_UPDATE]
        print("[Warning] No Episode Update Time Data")
        return -1

    def get_agent_multi_env_eval_rewards(self, agent_ID, eval_event_num):
        """Gets the dictionary of rewards per environment for a specific evaluation event."""
        agent_episode = self._get_agent_episode_dict(agent_ID, eval_event_num)
        if KEY_AGENT_MULTI_ENV_EVAL_REWARDS in agent_episode:
            return agent_episode[KEY_AGENT_MULTI_ENV_EVAL_REWARDS]
        # print(f"[Warning] No Multi-Env Eval Reward Data for ep {eval_event_num}") # Optional Warning
        return None # Return None if data for this event doesn't exist