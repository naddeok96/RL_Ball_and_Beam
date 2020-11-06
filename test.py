from env.BeamEnv import BeamEnv

env = BeamEnv()
obs = env.reset()

env.sample_freq = 10

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done = env.step(obs, action, render=True)
    print(obs)
