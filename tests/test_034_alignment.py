import math
import pytest
from importlib.machinery import SourceFileLoader
import os

# Unique module name required by rules
impl_path = os.path.join(os.path.dirname(__file__), '..', 'modules', '034-preference-alignment-rlhf-ppo', '04-implementation.py')
module_034_impl = SourceFileLoader("module_034_impl", impl_path).load_module()

# ---------------------------------------------------------
# Category 1: Bradley-Terry Probability (4 tests)
# ---------------------------------------------------------

def test_bt_equal_reward():
    rm = module_034_impl.BradleyTerryRewardModel()
    # Weights empty, score 0 for both
    p = rm.get_probability("a", "b")
    assert math.isclose(p, 0.5)

def test_bt_positive_diff():
    rm = module_034_impl.BradleyTerryRewardModel()
    rm.weights["good"] = 2.0
    rm.weights["bad"] = 0.0
    p = rm.get_probability("good", "bad")
    # sigmoid(2) ≈ 0.8808
    assert p > 0.88

def test_bt_negative_diff():
    rm = module_034_impl.BradleyTerryRewardModel()
    rm.weights["good"] = 0.0
    rm.weights["bad"] = 2.0
    p = rm.get_probability("good", "bad")
    assert p < 0.12

def test_bt_training_step():
    rm = module_034_impl.BradleyTerryRewardModel()
    rm.train_step("win", "lose")
    assert rm.weights["win"] > 0.0
    assert rm.weights["lose"] < 0.0

# ---------------------------------------------------------
# Category 2: KL Divergence Math (4 tests)
# ---------------------------------------------------------

def test_kl_zero_diff():
    kl = module_034_impl.KLController(beta=0.1)
    penalty = kl.compute_penalty(-1.0, -1.0)
    assert math.isclose(penalty, 0.0)

def test_kl_positive_drift():
    kl = module_034_impl.KLController(beta=0.5)
    penalty = kl.compute_penalty(-1.0, -2.0)
    # 0.5 * (-1.0 - -2.0) = 0.5 * 1.0 = 0.5
    assert math.isclose(penalty, 0.5)

def test_kl_negative_drift():
    kl = module_034_impl.KLController(beta=0.1)
    penalty = kl.compute_penalty(-2.0, -1.0)
    assert math.isclose(penalty, -0.1)

def test_kl_beta_scaling():
    kl1 = module_034_impl.KLController(beta=0.1)
    kl2 = module_034_impl.KLController(beta=1.0)
    p1 = kl1.compute_penalty(-1.0, -2.0)
    p2 = kl2.compute_penalty(-1.0, -2.0)
    assert p2 == p1 * 10.0

# ---------------------------------------------------------
# Category 3: DPO Loss Function (4 tests)
# ---------------------------------------------------------

def test_dpo_loss_equal():
    dpo = module_034_impl.DPOLossCalculator(beta=0.1)
    loss = dpo.compute_loss(-1.0, -1.0, -1.0, -1.0)
    # diff = 0, log(1 + exp(0)) = log(2) ≈ 0.693
    assert math.isclose(loss, math.log(2.0))

def test_dpo_loss_improvement():
    dpo = module_034_impl.DPOLossCalculator(beta=0.1)
    loss_start = dpo.compute_loss(-1.0, -1.0, -1.0, -1.0)
    loss_better = dpo.compute_loss(-0.5, -1.0, -2.0, -1.0)
    assert loss_better < loss_start

def test_dpo_loss_worsening():
    dpo = module_034_impl.DPOLossCalculator(beta=0.1)
    loss_start = dpo.compute_loss(-1.0, -1.0, -1.0, -1.0)
    loss_worse = dpo.compute_loss(-2.0, -1.0, -0.5, -1.0)
    assert loss_worse > loss_start

def test_dpo_loss_beta_sensitivity():
    dpo1 = module_034_impl.DPOLossCalculator(beta=0.1)
    dpo2 = module_034_impl.DPOLossCalculator(beta=1.0)
    l1 = dpo1.compute_loss(-0.5, -1.0, -2.0, -1.0)
    l2 = dpo2.compute_loss(-0.5, -1.0, -2.0, -1.0)
    assert l1 != l2

# ---------------------------------------------------------
# Category 4: Preference Dataset & PPO Clipping (4 tests)
# ---------------------------------------------------------

def test_dataset_length():
    ds = module_034_impl.PreferenceDataset([("p", "w", "l")] * 5)
    assert len(ds) == 5

def test_dataset_iteration():
    ds = module_034_impl.PreferenceDataset([("p1", "w1", "l1"), ("p2", "w2", "l2")])
    items = list(ds)
    assert len(items) == 2
    assert items[0][1] == "w1"

def test_ppo_no_clip():
    ppo = module_034_impl.PPOTrainer(None, None, clip_ratio=0.2)
    # ratio = exp(0) = 1.0, advantage = 1.0
    loss = ppo.compute_loss(-1.0, -1.0, 1.0)
    # min(1.0, 1.0) = 1.0, return -1.0
    assert math.isclose(loss, -1.0)

def test_ppo_clip():
    ppo = module_034_impl.PPOTrainer(None, None, clip_ratio=0.2)
    # old=-1.0, new=1.0 -> ratio = exp(2.0) ≈ 7.38
    loss = ppo.compute_loss(-1.0, 1.0, 1.0)
    # clip(7.38, 0.8, 1.2) = 1.2
    # obj = min(7.38, 1.2) = 1.2 -> return -1.2
    assert math.isclose(loss, -1.2)
