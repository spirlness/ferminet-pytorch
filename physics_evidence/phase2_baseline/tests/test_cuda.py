"""Run numerical regressions on CUDA, including CPU/GPU energy agreement.

CUDA tests are explicitly skipped when the installed torch is CPU-only.
Run with a CUDA-enabled interpreter: python -m unittest test_cuda -v.
"""

import unittest
from unittest.mock import patch

import torch

import tests.test_determinants as determinant_tests
import tests.test_p2_regressions as p2_tests
import tests.test_envelope as envelope_tests
from ferminet_pytorch.hamiltonian import compute_local_energy, MolecularSystem
from ferminet_pytorch.model import FermiNet
from ferminet_pytorch.train_vmc import build_h2_system, build_lih_system, train_vmc


class _CudaContext:
    """Ensure inherited tests create both coordinates and model parameters on GPU."""

    def setUp(self) -> None:
        context = torch.device("cuda")
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        self.addCleanup(torch.cuda.synchronize)
        super().setUp()
        self.assertEqual(torch.zeros(1).device.type, "cuda")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA-enabled torch and GPU required")
class TestCudaDeterminants(_CudaContext, determinant_tests.TestSingularDeterminants):
    """First/second/parameter derivatives, rank deficiency, and tiny amplitudes."""


@unittest.skipUnless(torch.cuda.is_available(), "CUDA-enabled torch and GPU required")
class TestCudaEnvelope(_CudaContext, envelope_tests.TestNuclearEnvelope):
    """Unified envelope values and coordinate/parameter derivatives on CUDA."""


@unittest.skipUnless(torch.cuda.is_available(), "CUDA-enabled torch and GPU required")
class TestCudaSpin(_CudaContext, p2_tests.TestSpinProjectionRegressions):
    """S^2 eigenstates, idempotence, antisymmetry, and custom permutations."""


@unittest.skipUnless(torch.cuda.is_available(), "CUDA-enabled torch and GPU required")
class TestCudaTripletCusp(_CudaContext, p2_tests.TestTripletCuspRegressions):
    """Triplet cusp slopes and bounded coalescence energies."""


@unittest.skipUnless(torch.cuda.is_available(), "CUDA-enabled torch and GPU required")
class TestCudaInvalidEnergy(_CudaContext, p2_tests.TestInvalidEnergyRegressions):
    def test_training_aborts_before_optimizer_step(self) -> None:
        torch.manual_seed(42)
        system = MolecularSystem(torch.zeros(1, 3), torch.ones(1), 1, 0)
        model = FermiNet(system.nuclear_coords, system.nuclear_charges, 1, 0,
                         num_determinants=2, hidden_dim_one=8, hidden_dim_two=4, num_layers=1)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        invalid = torch.full((4,), float("nan"))
        with patch("ferminet_pytorch.loss.compute_local_energy", return_value=(invalid,) * 3):
            with patch("torch.optim.Adam.step") as step:
                with self.assertRaisesRegex(FloatingPointError, "No finite"):
                    train_vmc(system=system, model=model, num_walkers=4, num_epochs=1,
                              burn_in_steps=0, mcmc_steps_per_epoch=1, device="cuda", verbose=False)
                step.assert_not_called()
        for old, new in zip(before, model.parameters()):
            self.assertEqual(new.device.type, "cuda")
            torch.testing.assert_close(old, new, atol=0, rtol=0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA-enabled torch and GPU required")
class TestCudaAgreementAndTraining(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.addCleanup(torch.cuda.synchronize)

    def test_cpu_cuda_local_energy_and_coordinate_gradients_agree(self) -> None:
        for builder in (build_h2_system, build_lih_system):
            with self.subTest(system=builder.__name__):
                system_cpu = builder(dtype=torch.float64)
                model_cpu = FermiNet(system_cpu.nuclear_coords, system_cpu.nuclear_charges,
                                     system_cpu.n_up, system_cpu.n_down, num_determinants=2,
                                     hidden_dim_one=8, hidden_dim_two=4, num_layers=1,
                                     use_cusp_jastrow=True, spin_adapted=True).double()
                system_gpu = builder(device=torch.device("cuda"), dtype=torch.float64)
                model_gpu = FermiNet(system_gpu.nuclear_coords, system_gpu.nuclear_charges,
                                     system_gpu.n_up, system_gpu.n_down, num_determinants=2,
                                     hidden_dim_one=8, hidden_dim_two=4, num_layers=1,
                                     use_cusp_jastrow=True, spin_adapted=True).to("cuda", torch.float64)
                model_gpu.load_state_dict(model_cpu.state_dict())
                r_cpu = torch.randn(2, system_cpu.total_electrons, 3,
                                    dtype=torch.float64, requires_grad=True)
                r_gpu = r_cpu.detach().to("cuda").requires_grad_()
                log_cpu, sign_cpu = model_cpu.forward_log_psi(r_cpu)
                log_gpu, sign_gpu = model_gpu.forward_log_psi(r_gpu)
                torch.testing.assert_close(log_gpu.cpu(), log_cpu, atol=1e-9, rtol=1e-9)
                torch.testing.assert_close(sign_gpu.cpu(), sign_cpu)
                grad_cpu = torch.autograd.grad(log_cpu.sum(), r_cpu)[0]
                grad_gpu = torch.autograd.grad(log_gpu.sum(), r_gpu)[0]
                torch.testing.assert_close(grad_gpu.cpu(), grad_cpu, atol=1e-8, rtol=1e-8)
                for gpu, cpu in zip(compute_local_energy(model_gpu, system_gpu, r_gpu),
                                    compute_local_energy(model_cpu, system_cpu, r_cpu)):
                    self.assertEqual(gpu.device.type, "cuda")
                    self.assertTrue(torch.isfinite(gpu).all())
                    torch.testing.assert_close(gpu.cpu(), cpu, atol=1e-7, rtol=1e-7)

    def test_h2_lih_training_updates_cuda_parameters(self) -> None:
        for builder, move_mode in ((build_h2_system, "one_by_one"), (build_lih_system, "all")):
            with self.subTest(system=builder.__name__, move_mode=move_mode):
                system = builder()
                model = FermiNet(system.nuclear_coords, system.nuclear_charges,
                                 system.n_up, system.n_down, num_determinants=2,
                                 hidden_dim_one=8, hidden_dim_two=4, num_layers=1,
                                 use_cusp_jastrow=True, spin_adapted=True).cuda()
                before = [parameter.detach().clone() for parameter in model.parameters()]
                result = train_vmc(system=system, model=model, num_walkers=16, num_epochs=3,
                                   burn_in_steps=5, mcmc_steps_per_epoch=2,
                                   move_mode=move_mode, device="cuda", verbose=False)
                self.assertTrue(all(parameter.device.type == "cuda" for parameter in model.parameters()))
                self.assertTrue(all(buffer.device.type == "cuda" for buffer in model.buffers()))
                self.assertEqual(result["sampler"].walkers.device.type, "cuda")
                self.assertEqual(result["history"]["num_valid"], [16] * 3)
                self.assertEqual(result["history"]["num_invalid"], [0] * 3)
                self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, model.parameters())))
                self.assertTrue(all(torch.isfinite(parameter).all() for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
