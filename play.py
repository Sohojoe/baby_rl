from baby_rl import *

def run_steps(agent):
    config = agent.config
    agent_name = agent.__class__.__name__
    t0 = time.time()

    eval_res = agent.eval_episodes()
    eval_score = eval_res['episodic_return_test']

    print('Average score:', eval_score)

def a2c_continuous(**kwargs):
    generate_tag(kwargs)
    # kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)


# TD3
def td3_continuous(**kwargs):
    generate_tag(kwargs)
    # kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)
    config.task_fn = lambda: Task(config.game, config.num_workers, marathon_envs=True)
    config.eval_env = Task(config.game, 3, marathon_envs=True, inference=True)
    config.network_fn = lambda: TD3Net(
        config.action_dim,
        actor_body_fn=lambda: FCBody(config.state_dim, (400, 300), gate=F.relu),
        critic_body_fn=lambda: FCBody(
            config.state_dim+config.action_dim, (400, 300), gate=F.relu),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3))

    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=config.mini_batch_size)
    config.discount = 0.99
    config.random_process_fn = lambda: GaussianProcess(
        size=(config.action_dim,), std=LinearSchedule(0.1))
    config.td3_noise = 0.2
    config.td3_noise_clip = 0.5
    config.td3_delay = 2
    config.target_network_mix = 5e-3    
    agent = TD3Agent(config)
    agent.load('examples/'+agent_name)
    run_steps(agent)


if __name__ == '__main__':
    mkdir('log')
    set_one_thread()
    random_seed()
    select_device(-1)
    # select_device(0)

    # game, agent_name = 'Hopper-v0', 'TD3Agent-Hopper-v0-target_score_500-run-0-700000'
    # game, agent_name = 'TerrainMarathonMan-v0', 'TD3Agent-TerrainMarathonMan-v0-target_score_500-run-0-5000000'
    # game, agent_name = 'TerrainMarathonMan-v0', 'TD3Agent-TerrainMarathonMan-v0-target_score_500-run-0-12500000'
    game, agent_name = 'TerrainMarathonMan-v0', 'TD3Agent-TerrainMarathonMan-v0-target_score_500-run-0-15300000'
    # a2c_continuous(game=game)
    # ppo_continuous(game=game)
    # ddpg_continuous(game=game)
    td3_continuous(game=game, agent_name=agent_name)