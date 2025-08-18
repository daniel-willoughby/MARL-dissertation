from pettingzoo.classic import tictactoe_v3
import numpy as np
import tf_keras


from tf_keras.models import Sequential
from tf_keras.layers import Flatten , Dense
from tf_keras.optimizers import Adam

from rl.agents import DQNAgent  # uses keras-rl2 or keras-rl


env = tictactoe_v3.env()
env.reset()
count = 0


actions = 9
states = 3
#actions = env.action_space(agent= "player_1").(mask = "action_mask").shape
# states = env.observation_space(agent= "player_1" )

print(states)
print(actions)


model = Sequential()
model.add(Flatten(input_shape=(1, states)))
model.add(Dense(24, activation="relu"))
model.add(Dense(24, activation="relu"))
model.add(Dense(actions, activation="linear"))

agent = DQNAgent(
    model=model,
    memory=SequentialMemory(limit=50000, window_length=1),
    policy=BoltzmannQPolicy(),
    nb_actions=actions,
    nb_steps_warmup=10,
    target_model_update=0.01
)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    print(agent)

    if termination or truncation:
        # print("looped1")
        action = 1
    else:
        # print("looped2")
        action = env.action_space(agent).sample()  # This is where you would insert your agent's policy


    
    #added this comment for push trial 2 and 3


    # added this comment for pull trial 1




    

    env.step(action)

    env = tictactoe_v3.env(render_mode="human")
    env.reset()
    for _ in range(100):
        env.render()
        #env.step(env.action_space(env.agent_selection).sample())
        env.step(count)
        count += 1
        # print("looped3")
    env.close()
    print("loop complete")

