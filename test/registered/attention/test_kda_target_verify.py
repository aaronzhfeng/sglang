"""Smoke test for TritonKDAKernel.target_verify, added in commit 0ef5cc929
to implement the KDA spec-decoding verify path.

Tests three properties:
  1. target_verify method exists with the expected signature
  2. Runs without error on Kimi-Linear-shaped inputs (num_heads=8, head_dim=128)
  3. Returns a finite tensor of shape [1, T, num_heads, head_dim]

Skips buffer-write correctness because intermediate_states_buffer layout depends
on MambaPool.SpeculativeState pool allocation, which is exercised end-to-end by
the DFlash worker (a separate test fixture).
"""
import inspect
import unittest

import torch

from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from sglang.srt.utils.common import get_device
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=6, suite="stage-b-test-1-gpu-large")


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestKDATargetVerifySignature(unittest.TestCase):
    """Catches accidental signature breaks in TritonKDAKernel.target_verify."""

    def test_method_exists(self):
        kernel = TritonKDAKernel()
        self.assertTrue(
            hasattr(kernel, "target_verify"),
            "TritonKDAKernel must expose target_verify after KDA verify-path patch",
        )

    def test_signature_has_spec_decoding_params(self):
        kernel = TritonKDAKernel()
        sig = inspect.signature(kernel.target_verify)
        expected = {
            "A_log",
            "dt_bias",
            "q",
            "k",
            "v",
            "a",
            "b",
            "ssm_states",
            "cache_indices",
            "query_start_loc",
            "intermediate_states_buffer",
            "intermediate_state_indices",
            "cache_steps",
            "retrieve_parent_token",
        }
        params = set(sig.parameters)
        missing = expected - params
        self.assertEqual(missing, set(), f"target_verify missing params: {missing}")


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestKDATargetVerifyRuns(unittest.TestCase):
    """Smoke: call target_verify on canonical shapes, assert finite output."""

    def setUp(self):
        self.device = get_device()
        self.batch = 2
        self.cache_steps = 4  # draft tokens per sequence
        self.token_num = self.batch * self.cache_steps
        self.num_heads = 8
        self.head_dim = 128
        self.cache_len = 16

        self.query_start_loc = torch.tensor(
            [0, self.cache_steps, self.token_num],
            dtype=torch.int32,
            device=self.device,
        )
        self.cache_indices = torch.tensor([3, 7], dtype=torch.int32, device=self.device)
        self.intermediate_state_indices = torch.tensor(
            [0, 1], dtype=torch.int32, device=self.device
        )

        H, D = self.num_heads, self.head_dim
        self.A_log = torch.randn(
            1, 1, H, 1, dtype=torch.float32, device=self.device
        )
        self.dt_bias = torch.randn(
            H * D, dtype=torch.bfloat16, device=self.device
        )
        self.q = torch.randn(
            1, self.token_num, H, D, dtype=torch.bfloat16, device=self.device
        )
        self.k = torch.randn(
            1, self.token_num, H, D, dtype=torch.bfloat16, device=self.device
        )
        self.v = torch.randn(
            1, self.token_num, H, D, dtype=torch.bfloat16, device=self.device
        )
        # KDA gate: channel-wise [1, T, H*D]
        self.a = torch.randn(
            1, self.token_num, H * D, dtype=torch.bfloat16, device=self.device
        )
        # KDA beta: per-head [1, T, H]
        self.beta = torch.randn(
            1, self.token_num, H, dtype=torch.bfloat16, device=self.device
        )

        self.ssm_states = torch.zeros(
            self.cache_len, H, D, D, dtype=torch.float32, device=self.device
        )
        # intermediate_states_buffer layout mirrors GDN's verify pool:
        # [intermediate_pool_size, cache_steps, H, D, D]
        self.intermediate_states_buffer = torch.zeros(
            2, self.cache_steps, H, D, D, dtype=torch.float32, device=self.device
        )
        # retrieve_parent_token = None disables the eagle tree branch
        # (linear acceptance, no per-step parent rewriting)
        self.retrieve_parent_token = None

    def test_runs_and_output_shape(self):
        kernel = TritonKDAKernel()
        out = kernel.target_verify(
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            q=self.q,
            k=self.k,
            v=self.v,
            a=self.a,
            b=self.beta,
            ssm_states=self.ssm_states,
            cache_indices=self.cache_indices,
            query_start_loc=self.query_start_loc,
            intermediate_states_buffer=self.intermediate_states_buffer,
            intermediate_state_indices=self.intermediate_state_indices,
            cache_steps=self.cache_steps,
            retrieve_parent_token=self.retrieve_parent_token,
        )
        self.assertEqual(
            out.shape,
            (1, self.token_num, self.num_heads, self.head_dim),
            f"unexpected output shape: {out.shape}",
        )
        self.assertTrue(torch.isfinite(out).all(), "output has non-finite values")


if __name__ == "__main__":
    unittest.main()
