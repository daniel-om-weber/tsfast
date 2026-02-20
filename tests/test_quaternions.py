"""Tests for tsfast.quaternions module."""
import pytest
import torch
import numpy as np


class TestQuaternionMath:
    def test_multiply_quat_identity(self):
        from tsfast.quaternions import multiplyQuat
        q = torch.tensor([[0.5, 0.5, 0.5, 0.5]])  # unit quaternion
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        result = multiplyQuat(q, identity)
        assert torch.allclose(result, q, atol=1e-5)

    def test_conj_quat_property(self):
        from tsfast.quaternions import multiplyQuat, conjQuat, norm_quaternion
        q = norm_quaternion(torch.rand(4, 4))
        result = multiplyQuat(q, conjQuat(q))
        # q * conj(q) should give [1, 0, 0, 0] for unit quaternions
        assert torch.allclose(result[..., 1:], torch.zeros(4, 3), atol=1e-5)
        assert torch.allclose(result[..., 0], torch.ones(4), atol=1e-5)

    def test_relative_quat_same_is_identity(self):
        from tsfast.quaternions import relativeQuat, norm_quaternion
        q = norm_quaternion(torch.rand(4, 4))
        result = relativeQuat(q, q)
        assert torch.allclose(result[..., 0].abs(), torch.ones(4), atol=1e-5)
        assert torch.allclose(result[..., 1:].abs(), torch.zeros(4, 3), atol=1e-4)

    def test_norm_quaternion_unit(self):
        from tsfast.quaternions import norm_quaternion
        q = torch.rand(10, 4) * 5  # non-unit
        normed = norm_quaternion(q)
        norms = normed.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(10), atol=1e-5)

    def test_inclination_angle_same_is_zero(self):
        from tsfast.quaternions import inclinationAngle, norm_quaternion
        # Use plain tensors to avoid torch.compile issues with TensorBase subclasses
        q = norm_quaternion(torch.rand(4, 100, 4)).as_subclass(torch.Tensor)
        angle = inclinationAngle(q, q)
        # Float32 precision limits: acos near 1.0 has ~1e-3 error
        assert angle.abs().max().item() < 2e-3

    def test_relative_angle_same_is_zero(self):
        from tsfast.quaternions import relativeAngle, norm_quaternion
        q = norm_quaternion(torch.rand(4, 100, 4)).as_subclass(torch.Tensor)
        angle = relativeAngle(q, q)
        assert angle.abs().max().item() < 2e-3

    def test_rand_quat_unit_norm(self):
        from tsfast.quaternions import rand_quat
        for _ in range(10):
            q = rand_quat()
            assert q.norm().item() == pytest.approx(1.0, abs=1e-5)


class TestQuaternionLosses:
    def test_inclination_loss_same_zero(self):
        from tsfast.quaternions import inclination_loss, norm_quaternion
        q = norm_quaternion(torch.rand(4, 100, 4))
        loss = inclination_loss(q, q)
        assert loss.item() < 1e-4

    def test_angle_loss_same_zero(self):
        from tsfast.quaternions import angle_loss, norm_quaternion
        q = norm_quaternion(torch.rand(4, 100, 4))
        loss = angle_loss(q, q)
        assert loss.item() < 1e-4

    def test_rms_inclination_deg_positive(self):
        from tsfast.quaternions import rms_inclination_deg, norm_quaternion
        q1 = norm_quaternion(torch.rand(4, 100, 4))
        q2 = norm_quaternion(torch.rand(4, 100, 4))
        val = rms_inclination_deg(q1, q2)
        assert val.item() > 0


class TestQuaternionAugmentation:
    def test_augmentation_groups_function(self):
        from tsfast.quaternions import augmentation_groups
        groups = augmentation_groups([3, 4, 3])
        assert groups == [[0, 2], [3, 6], [7, 9]]

    def test_quaternion_augmentation_modifies_input(self):
        from tsfast.quaternions import QuaternionAugmentation, TensorQuaternionInclination, norm_quaternion
        tfm = QuaternionAugmentation(inp_groups=[[0, 3]])
        q_raw = norm_quaternion(torch.rand(100, 4))
        q = TensorQuaternionInclination(q_raw)
        q_orig = q.clone()
        # Use eager mode to avoid torch.compile issues with TensorBase
        with torch.compiler.set_stance("force_eager"):
            augmented = tfm(q, split_idx=0)
        assert augmented.shape == q_orig.shape
        # Values should change (rotation applied)
        assert not torch.allclose(augmented.as_subclass(torch.Tensor), q_orig.as_subclass(torch.Tensor), atol=1e-3)


class TestQuaternionDataPipeline:
    def test_hdf2quaternion_extraction(self, orientation_path):
        from tsfast.quaternions import HDF2Quaternion
        hdf_file = orientation_path / "experiment2_linear_medium_b0_v_results_myon.mat.hdf5"
        seq = HDF2Quaternion(["opt_a", "opt_b", "opt_c", "opt_d"], cached=False)
        result = seq(hdf_file)
        assert result.shape == (54095, 4)

    def test_quaternion_block_from_hdf(self):
        from tsfast.quaternions import QuaternionBlock
        block = QuaternionBlock.from_hdf(["opt_a", "opt_b", "opt_c", "opt_d"])
        assert block is not None
        assert hasattr(block, "type_tfms")
