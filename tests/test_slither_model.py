import numpy as np
import pytest

torch = pytest.importorskip("torch")

from snake_rl.slither.model import SlitherNet, SlitherPolicy, load_slither_policy, masked_logits  # noqa: E402


def test_shapes_and_config():
    net = SlitherNet()
    logits, value = net(torch.zeros(5, 263))
    assert logits.shape == (5, 18) and value.shape == (5,)
    assert net.config() == {"obs_dim": 263, "hidden": 256, "actions": 18}


def test_masked_logits_keeps_all_when_nothing_is_allowed():
    out = masked_logits(torch.zeros(2, 3), torch.tensor([[True, False, True], [False, False, False]]))
    assert out[0, 1] < -1e8 and (out[1] == 0).all()


def test_policy_respects_mask_and_leading_shape():
    torch.manual_seed(0)
    policy = SlitherPolicy(SlitherNet())
    obs = np.random.default_rng(0).random((4, 3, 263)).astype(np.float32)
    mask = np.ones((4, 3, 18), bool)
    mask[..., 1::2] = False
    actions = policy(obs, mask)
    assert actions.shape == (4, 3) and (actions % 2 == 0).all()


def test_save_and_load(tmp_path):
    net = SlitherNet(hidden=32)
    path = tmp_path / "net.pt"
    torch.save({"config": net.config(), "model": net.state_dict()}, path)
    loaded = load_slither_policy(path)
    x = torch.rand(3, 263)
    torch.testing.assert_close(loaded(x)[0], net(x)[0])
