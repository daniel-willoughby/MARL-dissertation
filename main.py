from pettingzoo.classic import tictactoe_v3

env = tictactoe_v3.env()
env.reset()
count = 0

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        print("looped1")
        action = 1
    else:
        print("looped2")
        action = env.action_space(agent).sample()  # This is where you would insert your agent's policy
    #added this comment for push trial 2 and 3

    env.step(action)

    env = tictactoe_v3.env(render_mode="human")
    env.reset()
    for _ in range(100):
        env.render()
        #env.step(env.action_space(env.agent_selection).sample())
        env.step(count)
        count += 1
        print("looped3")
    env.close()
    print("loop complete")
