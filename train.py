from baby_rl import *

# TD3
def td3_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    n_agents = 1
    # n_agents = 8

    # config.task_fn = lambda: Task(config.game, n_agents, marathon_envs=True, no_graphics=True)
    config.task_fn = lambda: Task(config.game, n_agents, marathon_envs=True)
    config.eval_env = Task(config.game, marathon_envs=True)
    config.max_steps = int(1e5)
    config.eval_interval = int(1e4)
    config.eval_episodes = 10

    config.network_fn = lambda: TD3Net(
        config.action_dim,
        actor_body_fn=lambda: FCBody(config.state_dim, (400, 300), gate=F.relu),
        critic_body_fn=lambda: FCBody(
            config.state_dim+config.action_dim, (400, 300), gate=F.relu),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3))

    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=100)
    config.discount = 0.99
    config.random_process_fn = lambda: GaussianProcess(
        size=(config.action_dim,), std=LinearSchedule(0.1))
    config.td3_noise = 0.2
    config.td3_noise_clip = 0.5
    config.td3_delay = 2
    config.warm_up = int(1e4)
    config.target_network_mix = 5e-3
    run_steps(TD3Agent(config))

if __name__ == '__main__':
    mkdir('log')
    mkdir('tf_log')
    set_one_thread()
    random_seed()
    # select_device(-1)
    select_device(0)

    game = 'Hopper-v0'
    # from marathon_envs.envs import MarathonEnvs
    # import pathlib
    # aa = pathlib.Path().absolute()
    # marathon_envs_path = os.path.join(aa,'envs', 'MarathonEnvs', 'Unity Environment.exe')
    # # envs = MarathonEnvs(game, 1)
    # envs = MarathonEnvs(game, 1, marathon_envs_path=marathon_envs_path)
    # game = 'Walker2d-v0'
    # game = 'Ant-v0'
    # game = 'MarathonMan-v0'
    # game = 'MarathonManSparse-v0'
    # a2c_continuous(game=game)
    # ppo_continuous(game=game)
    # ddpg_continuous(game=game)
    td3_continuous(game=game)