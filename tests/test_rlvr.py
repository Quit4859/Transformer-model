import pytest
import torch

from agentmodel.rlvr.grpo import (
    GroupBatch,
    clipped_policy_loss,
    dynamic_sampling,
    group_advantages,
    grpo_loss,
    masked_mean,
)
from agentmodel.rlvr.rewards import filter_zero_signal, grade_rollout, reward_spread


def test_reward_grader_rewards_tests_and_penalizes_steps():
    good = grade_rollout(tests_passed=True, valid_format=True, steps=0)
    wasteful = grade_rollout(tests_passed=True, valid_format=True, steps=20)
    assert good.total > wasteful.total
    assert wasteful.details["step_penalty"] == -0.2


def test_reward_grader_penalizes_invalid_format():
    formatted = grade_rollout(tests_passed=True, valid_format=True)
    malformed = grade_rollout(tests_passed=True, valid_format=False)
    assert formatted.total > malformed.total


def test_zero_signal_filter_drops_uniform_groups():
    assert filter_zero_signal([1.0, 1.0]) == []
    assert filter_zero_signal([0.0, 1.0]) == [0, 1]
    assert filter_zero_signal([]) == []


def test_reward_spread_is_zero_for_uniform_group():
    assert reward_spread([2.0, 2.0, 2.0]) == 0.0
    assert reward_spread([0.0, 2.0]) == pytest.approx(1.0)


def test_group_advantages_normalizes_within_group():
    advantages = group_advantages(torch.tensor([[0.0, 1.0, 2.0]]))
    assert advantages.mean().abs().item() < 1e-6
    assert advantages[0, 0] < advantages[0, 2]


def test_group_advantages_zero_for_uniform_group():
    advantages = group_advantages(torch.tensor([[2.0, 2.0]]))
    assert torch.allclose(advantages, torch.zeros_like(advantages))


def test_group_advantages_rejects_wrong_rank():
    with pytest.raises(ValueError):
        group_advantages(torch.tensor([0.0, 1.0]))


def test_clipped_policy_loss_bounds_ratio():
    advantages = torch.ones(1, 1)
    assert torch.isfinite(clipped_policy_loss(torch.zeros_like(advantages), advantages))
    huge = clipped_policy_loss(torch.full((1, 1), 5.0), advantages, clip_ratio=0.2)
    assert huge.item() == pytest.approx(-1.2, abs=1e-4)


def test_masked_mean_ignores_padding():
    values = torch.tensor([[1.0, 5.0]])
    mask = torch.tensor([[1.0, 0.0]])
    assert masked_mean(values, mask).item() == pytest.approx(1.0)


def test_masked_mean_of_empty_mask_is_zero():
    assert masked_mean(torch.ones(2, 3), torch.zeros(2, 3)).item() == 0.0


def test_grpo_loss_falls_when_policy_matches_high_reward_rollout():
    mask = torch.ones(2, 3)
    rewards = torch.tensor([0.0, 1.0])
    matching = GroupBatch(
        log_probs=torch.zeros(2, 3), old_log_probs=torch.zeros(2, 3), rewards=rewards, mask=mask
    )
    inverted = GroupBatch(
        log_probs=torch.tensor([[2.0, 2.0, 2.0], [-2.0, -2.0, -2.0]]),
        old_log_probs=torch.zeros(2, 3),
        rewards=rewards,
        mask=mask,
    )
    assert grpo_loss(matching).item() == pytest.approx(0.0, abs=1e-4)
    assert grpo_loss(inverted).item() > grpo_loss(matching).item()


def test_dynamic_sampling_drops_uniform_prompts():
    kept, spreads = dynamic_sampling([[1.0, 1.0, 1.0], [0.0, 1.0]])
    assert kept == [1]
    assert spreads[0] > 0


def test_dynamic_sampling_reports_empty_for_all_uniform():
    assert dynamic_sampling([[0.0, 0.0], [1.0, 1.0]]) == ([], [])