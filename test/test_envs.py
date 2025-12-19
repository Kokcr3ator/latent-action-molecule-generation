import torch

from interdiff.envs import MoleculeGenerationEnv, tree_map, Timestep


def test_tree_map():
    def add(x, y):
        return x + y

    t1 = Timestep(
        observation=torch.tensor([1, 2]),
        t=torch.tensor([0]),
        reward=torch.tensor([0.0]),
        step_type=torch.tensor([0], dtype=torch.uint8),
        rng=torch.tensor([42]),
        info={"returns": torch.tensor([0.0])},
    )

    t2 = Timestep(
        observation=torch.tensor([3, 4]),
        t=torch.tensor([1]),
        reward=torch.tensor([1.0]),
        step_type=torch.tensor([1], dtype=torch.uint8),
        rng=torch.tensor([43]),
        info={"returns": torch.tensor([1.0])},
    )

    result = tree_map(add, t1, t2)

    assert torch.equal(result.observation, torch.tensor([4, 6]))
    assert torch.equal(result.t, torch.tensor([1]))
    assert torch.equal(result.reward, torch.tensor([1.0]))
    assert torch.equal(result.step_type, torch.tensor([1], dtype=torch.uint8))
    assert torch.equal(result.rng, torch.tensor([85]))
    assert torch.equal(result.info["returns"], torch.tensor([1.0]))


def test_molecule_generation_env():
    num_envs = 5
    max_steps = 10
    discount = 0.99

    env = MoleculeGenerationEnv(
        num_envs=num_envs, max_steps=max_steps, discount=discount, random_start=False
    )

    # get env dimensions
    action_dim = 1
    observation_dim = env.context_length

    seeds = torch.arange(100, 100 + num_envs)
    timestep = env.reset(seeds)

    print(timestep)
    assert timestep.observation.shape == (num_envs, observation_dim)
    assert torch.all(timestep.t == 0)
    assert torch.all(timestep.reward == 0.0)
    assert torch.all(timestep.step_type == 0)
    assert torch.equal(timestep.rng, seeds)
    assert torch.all(timestep.info["returns"] == 0.0)

    action = torch.randint(0, action_dim, (num_envs,))
    next_timestep = env.step(timestep, action)

    assert next_timestep.observation.shape == (num_envs, observation_dim)
    assert torch.all(next_timestep.t == 1)
    assert next_timestep.reward.shape == (num_envs,)
    assert next_timestep.step_type.shape == (num_envs,)
    assert torch.equal(next_timestep.rng, seeds + 1)
    assert next_timestep.info["returns"].shape == (num_envs,)


def test_molecule_generation_env_timeout():
    num_envs = 3
    max_steps = 5
    discount = 0.95

    env = MoleculeGenerationEnv(
        num_envs=num_envs, max_steps=max_steps, discount=discount
    )

    seeds = torch.arange(200, 200 + num_envs)
    timestep = env.reset(seeds)

    for _ in range(max_steps):
        action = torch.randint(2, 10, (num_envs,))
        timestep = env._step(timestep, action)

    # After max_steps, the environment should timeout
    assert torch.all(timestep.step_type == 2)  # Assuming 2 indicates timeout
    for ret in timestep.info["returns"]:
        assert ret >= 0.0  # Ensure returns are non-negative
    assert torch.all(timestep.t == max_steps)
