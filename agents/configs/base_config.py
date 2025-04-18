class LLMRLAgentConfig:

  # whatever data this particular config needs can be passed here
  def __init__(self):
    pass

  
  # whatever data this particular config needs at each time step can be passed here
  # config should convert "text observation" and other meta data into the prompt used for the agent
  # we will use duck typing here to have different configs which require different inputs
  def generate_prompt(self):
    return ""