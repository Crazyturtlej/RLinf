#!/usr/bin/env python3
"""Test script for RoboSimStudioEnv wrapper.

This script tests the basic functionality of our RoboSimStudio integration
with RLinf before attempting full training.
"""

import sys
import numpy as np
import torch
from omegaconf import OmegaConf

# Import our wrapper
from rlinf.envs.sim.robosimstudio import RoboSimStudioEnv


def create_test_config():
    """Create a minimal test configuration."""
    cfg = {
        # Basic settings
        "seed": 42,
        "auto_reset": True,
        "ignore_terminations": False,
        "use_rel_reward": True,
        "group_size": 1,
        "use_fixed_reset_state_ids": False,

        # RoboSimStudio specific
        "task_name": "usbc_insert/insert_easy",  # USB-C cable insertion task
        "max_steps_per_rollout_epoch": 500,
        "max_steps": 500,
        "renderer": "",  # No rendering for quick test
        "render_every": 1,
        "main_camera": "camera_0",
        "wrist_camera": "",

        # Camera settings
        "init_params": {
            "camera_heights": 256,
            "camera_widths": 256,
        },

        # Video config (disabled for test)
        "video_cfg": None,
    }

    return OmegaConf.create(cfg)


def test_env_creation():
    """Test 1: Environment creation."""
    print("\n" + "="*60)
    print("TEST 1: Environment Creation")
    print("="*60)

    cfg = create_test_config()

    try:
        env = RoboSimStudioEnv(
            cfg=cfg,
            num_envs=2,
            seed_offset=0,
            total_num_processes=1,
            worker_info={"rank": 0, "world_size": 1},
            record_metrics=True,
        )
        print("✓ Environment created successfully")
        print(f"  - Number of environments: {env.num_envs}")
        print(f"  - Task: {env.task_name}")
        print(f"  - Action space: {env.action_space}")
        print(f"  - State dimension: {env._state_dim}")
        return env
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_reset(env):
    """Test 2: Environment reset."""
    print("\n" + "="*60)
    print("TEST 2: Environment Reset")
    print("="*60)

    try:
        obs, info = env.reset()

        print("✓ Reset successful")
        print(f"\nObservation keys: {list(obs.keys())}")

        # Check states
        if "states" in obs:
            print(f"  - states shape: {obs['states'].shape}")
            print(f"  - states dtype: {obs['states'].dtype}")
            print(f"  - states range: [{obs['states'].min():.3f}, {obs['states'].max():.3f}]")

        # Check main images
        if "main_images" in obs:
            print(f"  - main_images shape: {obs['main_images'].shape}")
            print(f"  - main_images dtype: {obs['main_images'].dtype}")
            print(f"  - main_images range: [{obs['main_images'].min()}, {obs['main_images'].max()}]")

        # Check wrist images
        if "wrist_images" in obs:
            print(f"  - wrist_images shape: {obs['wrist_images'].shape}")
            print(f"  - wrist_images dtype: {obs['wrist_images'].dtype}")

        # Check task descriptions
        if "task_descriptions" in obs:
            print(f"  - task_descriptions: {obs['task_descriptions'][:1]}...")  # Print first one

        print(f"\nInfo keys: {list(info.keys())}")

        return obs

    except Exception as e:
        print(f"✗ Reset failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_step(env):
    """Test 3: Environment step."""
    print("\n" + "="*60)
    print("TEST 3: Environment Step")
    print("="*60)

    try:
        # Create random actions
        action_shape = (env.num_envs, env.action_space.shape[0])
        actions = np.random.uniform(-0.1, 0.1, size=action_shape).astype(np.float32)
        actions = torch.from_numpy(actions)

        print(f"Stepping with actions shape: {actions.shape}")

        obs, rewards, terminations, truncations, info = env.step(actions)

        print("✓ Step successful")
        print(f"  - Rewards: {rewards}")
        print(f"  - Terminations: {terminations}")
        print(f"  - Truncations: {truncations}")
        print(f"  - States shape: {obs['states'].shape}")
        print(f"  - Elapsed steps: {env.elapsed_steps}")

        # Check if metrics are being tracked
        if "episode" in info:
            print(f"\nEpisode metrics:")
            for key, value in info["episode"].items():
                if isinstance(value, torch.Tensor):
                    print(f"  - {key}: {value}")

        return True

    except Exception as e:
        print(f"✗ Step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_steps(env, num_steps=10):
    """Test 4: Multiple steps."""
    print("\n" + "="*60)
    print(f"TEST 4: Multiple Steps (n={num_steps})")
    print("="*60)

    try:
        for i in range(num_steps):
            action_shape = (env.num_envs, env.action_space.shape[0])
            actions = np.random.uniform(-0.1, 0.1, size=action_shape).astype(np.float32)
            actions = torch.from_numpy(actions)

            obs, rewards, terminations, truncations, info = env.step(actions)

            if i % 5 == 0:
                print(f"  Step {i}: rewards={rewards.numpy()}, done={terminations.any().item()}")

        print(f"✓ Completed {num_steps} steps successfully")
        print(f"  - Final elapsed steps: {env.elapsed_steps}")
        return True

    except Exception as e:
        print(f"✗ Multiple steps failed at step {i}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_observation_format_compatibility():
    """Test 5: Check observation format matches RLinf expectations."""
    print("\n" + "="*60)
    print("TEST 5: Observation Format Compatibility Check")
    print("="*60)

    cfg = create_test_config()
    env = RoboSimStudioEnv(
        cfg=cfg,
        num_envs=2,
        seed_offset=0,
        total_num_processes=1,
        worker_info={"rank": 0, "world_size": 1},
    )

    obs, _ = env.reset()

    # Expected format for OpenPi/VLA models
    expected_keys = ["states", "main_images", "wrist_images", "task_descriptions"]

    print("Checking observation format:")
    all_good = True

    for key in expected_keys:
        if key in obs:
            print(f"  ✓ {key}: present")

            # Check types
            if key in ["states", "main_images", "wrist_images"]:
                if isinstance(obs[key], torch.Tensor):
                    print(f"    - Type: torch.Tensor ✓")
                else:
                    print(f"    - Type: {type(obs[key])} (expected torch.Tensor)")
                    all_good = False

            # Check shapes
            if key == "states":
                expected_shape = (env.num_envs, env._state_dim)
                if obs[key].shape == expected_shape:
                    print(f"    - Shape: {obs[key].shape} ✓")
                else:
                    print(f"    - Shape: {obs[key].shape} (expected {expected_shape})")
                    all_good = False

            elif key in ["main_images", "wrist_images"]:
                expected_shape = (env.num_envs, 256, 256, 3)
                if obs[key].shape == expected_shape:
                    print(f"    - Shape: {obs[key].shape} ✓")
                else:
                    print(f"    - Shape: {obs[key].shape} (expected {expected_shape})")
                    all_good = False
        else:
            print(f"  ✗ {key}: missing")
            all_good = False

    if all_good:
        print("\n✓ All observation format checks passed")
    else:
        print("\n✗ Some observation format checks failed")

    env.close()
    return all_good


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# RoboSimStudio Environment Wrapper Test Suite")
    print("#"*60)

    results = []

    # Test 1: Create environment
    env = test_env_creation()
    results.append(("Environment Creation", env is not None))

    if env is None:
        print("\n✗ Cannot proceed with tests - environment creation failed")
        return False

    # Test 2: Reset
    obs = test_reset(env)
    results.append(("Reset", obs is not None))

    if obs is None:
        print("\n✗ Cannot proceed - reset failed")
        env.close()
        return False

    # Test 3: Single step
    step_success = test_step(env)
    results.append(("Single Step", step_success))

    if not step_success:
        print("\n✗ Cannot proceed - step failed")
        env.close()
        return False

    # Test 4: Multiple steps
    multi_step_success = test_multiple_steps(env, num_steps=10)
    results.append(("Multiple Steps", multi_step_success))

    # Clean up
    env.close()

    # Test 5: Observation format compatibility
    format_ok = test_observation_format_compatibility()
    results.append(("Observation Format", format_ok))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<40} {status}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("\n🎉 All tests passed! The wrapper is working correctly.")
        print("\nNext steps:")
        print("1. Create full training configuration")
        print("2. Test with actual OpenPi model")
        print("3. Run RL finetuning")
        return True
    else:
        print("\n⚠️  Some tests failed. Please fix the issues before proceeding.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
