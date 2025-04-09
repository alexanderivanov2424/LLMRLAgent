import textworld.gym
import gym
from textworld import EnvInfos

from textworld.generator import make_game, compile_game

options = textworld.GameOptions()
options.seeds = 1234
game = make_game(options)
# game.extras["more"] = "This is extra information."
gamefile = compile_game(game)

import gym
import textworld.gym
from textworld import EnvInfos

request_infos = EnvInfos(description=True, inventory=True, extras=["more"])
env_id = textworld.gym.register_game(gamefile, request_infos)
env = textworld.gym.make(env_id)
ob, infos = env.reset()
print(env.step("open chest"))
print(infos["description"])
print(infos["inventory"])


# request_infos = EnvInfos(description=True, inventory=True, extras=["more"])
# env_id = textworld.gym.register_game(
#     "tw_games/tw-simple-rDense+gDetailed+train-house-GP-oNKWsPmXirE9hpWp.json",
#     request_infos,
# )
# print(env_id)
# env = gym.make(env_id)
# ob, infos = env.reset()
# print(ob, infos)
# print(infos["extra.more"])
# print(infos["description"])
# print(infos["inventory"])
