"""
Runnable experiments for Preference Alignment (PPO vs DPO).
"""

from importlib.machinery import SourceFileLoader
impl = SourceFileLoader("module_034_impl", "modules/034-preference-alignment-rlhf-ppo/04-implementation.py").load_module()

BradleyTerryRewardModel = impl.BradleyTerryRewardModel
KLController = impl.KLController
PPOTrainer = impl.PPOTrainer
DPOLossCalculator = impl.DPOLossCalculator
PreferenceDataset = impl.PreferenceDataset

def run_experiment_1_bradley_terry():
    print("--- Experiment 1: Bradley-Terry Reward Model Fitting ---")
    dataset = PreferenceDataset([
        ("How to boil an egg?", "Boil for 7 minutes.", "Use a microwave for 10 minutes."),
        ("What is AI?", "Artificial Intelligence.", "Alien Invasion.")
    ])
    
    rm = BradleyTerryRewardModel()
    rm.learning_rate = 0.5
    
    for epoch in range(10):
        for prompt, yw, yl in dataset:
            rm.train_step(yw, yl)
            
    p_win_1 = rm.get_probability("Boil for 7 minutes.", "Use a microwave for 10 minutes.")
    print(f"P(win) for egg query after training: {p_win_1:.4f}")
    assert p_win_1 > 0.5, "Model failed to learn preference"

def run_experiment_2_dpo_loss():
    print("\n--- Experiment 2: DPO Loss Trajectory ---")
    dpo = DPOLossCalculator(beta=0.1)
    
    # Assume reference log probs are fixed
    ref_w, ref_l = -1.0, -1.0
    
    # Policy starts the same
    pol_w_start, pol_l_start = -1.0, -1.0
    loss_start = dpo.compute_loss(pol_w_start, ref_w, pol_l_start, ref_l)
    print(f"Initial DPO Loss: {loss_start:.4f}")
    
    # Policy improves (assigns higher prob to win, lower to lose)
    pol_w_end, pol_l_end = -0.5, -2.0
    loss_end = dpo.compute_loss(pol_w_end, ref_w, pol_l_end, ref_l)
    print(f"Final DPO Loss: {loss_end:.4f}")
    
    assert loss_end < loss_start, "Loss should decrease as policy aligns with preference"

def run_experiment_3_kl_penalty():
    print("\n--- Experiment 3: KL Penalty Effect ---")
    kl_low = KLController(beta=0.01)
    kl_high = KLController(beta=1.0)
    
    # Significant drift
    log_p_pol = -0.1
    log_p_ref = -3.0
    
    pen_low = kl_low.compute_penalty(log_p_pol, log_p_ref)
    pen_high = kl_high.compute_penalty(log_p_pol, log_p_ref)
    
    print(f"Penalty with beta=0.01: {pen_low:.4f}")
    print(f"Penalty with beta=1.0: {pen_high:.4f}")
    
    assert pen_high > pen_low, "Higher beta should impose higher penalty"

def run_experiment_4_ppo_clip():
    print("\n--- Experiment 4: PPO Clipped Objective ---")
    rm = BradleyTerryRewardModel()
    kl = KLController(beta=0.1)
    ppo = PPOTrainer(rm, kl, clip_ratio=0.2)
    
    adv = 1.0 # Positive advantage
    
    # Small ratio
    loss_small = ppo.compute_loss(-1.0, -0.9, adv)
    print(f"Loss with small ratio: {loss_small:.4f}")
    
    # Large ratio (clipped)
    loss_large = ppo.compute_loss(-1.0, 1.0, adv)
    print(f"Loss with large ratio (should be clipped): {loss_large:.4f}")
    
    assert loss_large >= -1.2, "Objective was not correctly clipped"

if __name__ == "__main__":
    run_experiment_1_bradley_terry()
    run_experiment_2_dpo_loss()
    run_experiment_3_kl_penalty()
    run_experiment_4_ppo_clip()
