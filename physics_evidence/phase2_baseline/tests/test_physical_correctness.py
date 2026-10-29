"""Independent physical oracles for the 2026-10 correctness audit."""
import unittest

import torch

from ferminet_pytorch.composable_model import BranchlessCofactorDeterminant, ComposableFermiNet
from ferminet_pytorch.geometry import GeometricContext
from ferminet_pytorch.hamiltonian import MolecularSystem, compute_local_energy
from ferminet_pytorch.kinetic_modern import KineticEnergyCalculator, KineticMode
from ferminet_pytorch.model import FermiNet, _CofactorDeterminant
from ferminet_pytorch.spin import SpinProjector
from ferminet_pytorch.vmc_driver import HamiltonianComposite, MetropolisSampler, VariationalState, VMCDriver


class TestPhysicalCorrectness(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.coords = torch.zeros(1, 3, dtype=torch.float64)
        self.charges = torch.ones(1, dtype=torch.float64)

    def net(self, cls=ComposableFermiNet, n_up=1, n_down=0, **kwargs):
        return cls(self.coords, self.charges, n_up, n_down,
                   num_determinants=2, hidden_dim_one=16,
                   hidden_dim_two=4, num_layers=1, **kwargs).double()

    def test_singular_branch_has_nonzero_first_and_second_derivatives(self):
        x = torch.tensor(0., dtype=torch.float64, requires_grad=True)
        matrices = torch.stack((x + x*x, 1 + x*0)).reshape(1, 2, 1, 1)
        log, _ = BranchlessCofactorDeterminant.compute(
            matrices, None, torch.ones(2, dtype=torch.float64), 1, 0, 2)
        first = torch.autograd.grad(log.sum(), x, create_graph=True)[0]
        second = torch.autograd.grad(first, x)[0]
        self.assertAlmostEqual(first.item(), 1., places=12)
        self.assertAlmostEqual(second.item(), 1., places=12)

    def test_large_singular_determinant_is_exact(self):
        for size in (7, 8, 10):
            m = torch.diag(torch.tensor([2.] * (size-1) + [0.], dtype=torch.float64))
            m.requires_grad_()
            value = _CofactorDeterminant.apply(m, torch.tensor(0., dtype=m.dtype))
            grad = torch.autograd.grad(value, m, create_graph=True)[0]
            expected = torch.zeros_like(m)
            expected[-1, -1] = 2. ** (size-1)
            torch.testing.assert_close(grad, expected, atol=1e-12, rtol=1e-12)

    def test_large_rank_deficiency_two_has_exact_mixed_derivative(self):
        for size in (7, 10):
            x = torch.tensor(0., dtype=torch.float64, requires_grad=True)
            y = torch.tensor(0., dtype=torch.float64, requires_grad=True)
            diagonal = torch.cat((torch.full((size-2,), 2., dtype=x.dtype), x[None], y[None]))
            value = _CofactorDeterminant.apply(torch.diag(diagonal), x.new_zeros(()))
            first = torch.autograd.grad(value, x, create_graph=True)[0]
            mixed = torch.autograd.grad(first, y)[0]
            self.assertEqual(first.item(), 0.)
            self.assertAlmostEqual(mixed.item(), 2.**(size-2), places=10)

    def test_context_created_under_no_grad_is_rebuilt(self):
        net = self.net()
        system = MolecularSystem(self.coords, self.charges, 1, 0)
        r = torch.tensor([[[.4, .2, .3]]], dtype=torch.float64, requires_grad=True)
        with torch.no_grad():
            context = GeometricContext.compute(r, self.coords, eps=net.eps)
        expected = compute_local_energy(net, system, r)[0].detach()
        actual = compute_local_energy(net, system, r, geom_ctx=context)[0].detach()
        torch.testing.assert_close(actual, expected)

    def test_feature_context_cannot_change_the_sampled_wavefunction(self):
        for cls in (FermiNet, ComposableFermiNet):
            net = self.net(cls)
            r = torch.tensor([[[.02, .01, .03]]], dtype=torch.float64)
            context = GeometricContext.compute(r, self.coords, eps=1.)
            expected = net.forward_log_psi(r)[0]
            actual = net.forward_log_psi(r, geom_ctx=context)[0]
            torch.testing.assert_close(actual, expected)
            with self.assertRaises(ValueError):
                VariationalState(net, MolecularSystem(self.coords,self.charges,1,0), eps=1.)

    def test_default_jastrow_has_finite_nuclear_limit(self):
        system = MolecularSystem(self.coords, self.charges, 1, 0)
        for cls in (FermiNet, ComposableFermiNet):
            net = self.net(cls, use_cusp_jastrow=True)
            linear = net.linear_orb_up if cls is FermiNet else net.projector_up.linear
            with torch.no_grad():
                linear.weight.zero_()
                linear.bias.fill_(1.)
            energies = []
            for distance in (1e-3, 1e-4, 1e-5):
                r = torch.tensor([[[distance, 0., 0.]]], dtype=torch.float64, requires_grad=True)
                log, _ = net.forward_log_psi(r)
                slope = torch.autograd.grad(log.sum(), r)[0][0, 0, 0]
                self.assertLess(abs(slope.item() + 1), 1e-4)
                energies.append(compute_local_energy(net, system, r)[0].item())
            self.assertLess(abs(energies[-1] + .5), 1e-3)

    def test_psiformer_pair_cusp_and_local_energy_limit(self):
        net = self.net(n_up=1, n_down=1, interaction_type='psiformer',
                       use_cusp_jastrow=True, spin_adapted=True)
        system = MolecularSystem(self.coords, self.charges, 1, 1)
        values = []
        for distance in (1e-3, 1e-4, 1e-5):
            t = torch.tensor(distance, dtype=torch.float64, requires_grad=True)
            offset = torch.stack((t/2, t*0, t*0))
            center = torch.tensor([.6, .7, .8], dtype=torch.float64)
            r = torch.stack((center+offset, center-offset)).unsqueeze(0)
            log, _ = net.forward_log_psi(r)
            slope = torch.autograd.grad(log.sum(), t)[0].item()
            values.append(compute_local_energy(net, system, r.detach().requires_grad_())[0].item())
        self.assertAlmostEqual(slope, .5, delta=1e-3)
        self.assertLess(abs(values[-1] - values[-2]), .05)

    def test_composable_triplet_uses_quarter_cusp(self):
        net = self.net(n_up=1, n_down=1, use_cusp_jastrow=True,
                       spin_projector=SpinProjector(1, 1, 'triplet'))
        self.assertEqual(net.jastrow.gamma_pairs.item(), .25)

    def test_potential_does_not_depend_on_feature_smoothing(self):
        system = MolecularSystem(self.coords, self.charges, 1, 0)
        r = torch.tensor([[[1e-7, 0., 0.]]], dtype=torch.float64)
        expected = system.compute_potential_energy(r)
        ctx = GeometricContext.compute(r, self.coords, eps=1e-6)
        torch.testing.assert_close(system.compute_potential_energy(r, geom_ctx=ctx), expected)

    def test_jvp_on_slogdet_and_model_with_context(self):
        def function(r):
            return torch.linalg.slogdet((1+r[:, 0, 0]**2).reshape(-1, 1, 1))[::-1]
        r = torch.tensor([[[.4, .2, .3]]], dtype=torch.float64, requires_grad=True)
        rev = KineticEnergyCalculator.compute_kinetic_energy(function, r)
        jvp = KineticEnergyCalculator.compute_kinetic_energy(function, r, mode=KineticMode.FORWARD_JVP)
        torch.testing.assert_close(jvp[0], rev[0], atol=1e-10, rtol=1e-10)
        for cls in (FermiNet, ComposableFermiNet):
            net = self.net(cls)
            system = MolecularSystem(self.coords, self.charges, 1, 0)
            r = r.detach().requires_grad_()
            ctx = GeometricContext.compute(r, self.coords, eps=net.eps)
            rev = compute_local_energy(net, system, r)[0]
            jvp = compute_local_energy(net, system, r, geom_ctx=ctx,
                                       kinetic_mode=KineticMode.FORWARD_JVP)[0]
            torch.testing.assert_close(jvp, rev, atol=1e-8, rtol=1e-7)

    def test_driver_refreshes_cache_after_each_update(self):
        net = self.net()
        system = MolecularSystem(self.coords, self.charges, 1, 0)
        state = VariationalState(net, system, num_walkers=8)
        sampler = MetropolisSampler(state)
        driver = VMCDriver(state, HamiltonianComposite(system), sampler,
                           torch.optim.SGD(net.parameters(), lr=.01), decorrelation_steps=1)
        for _ in range(2):
            driver.step()
            with torch.no_grad():
                expected = net.forward_log_psi(sampler.mcmc.walkers)[0]
            torch.testing.assert_close(sampler.mcmc.current_log_abs_psi, expected)


if __name__ == '__main__':
    unittest.main()
