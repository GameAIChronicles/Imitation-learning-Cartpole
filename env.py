import gymnasium as gym



env = gym.make('CartPole-v1',  render_mode='human')
print(env.observation_space)

episodes = 5
for episode in range(1, episodes + 1):
    state = env.reset()
    print(state)
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        a_state, reward, done, _, _ = env.step(action)
        score += reward
    print('Episodes: {}, Score: {}'.format(episode, score))
