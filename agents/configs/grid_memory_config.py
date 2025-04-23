from agents.configs.base_config import LLMRLAgentConfig


class GridMemoryConfig(LLMRLAgentConfig):

	# whatever data this particular config needs can be passed here
	def __init__(self):
		# FIXME
		self.context = []
		self.memory_length = 20
		self.importance_threshold = 0.5
		self.summarize_every_n = 10
		self.summarization_reward_threshold = 2.0

  
	# whatever data this particular config needs at each time step can be passed here
	# config should convert "text observation" and other meta data into the prompt used for the agent
	# we will use duck typing here to have different configs which require different inputs
	def generate_prompt(self):
		return ""