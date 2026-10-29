"""Physical invariants across architectures, with analytic reference operators."""
import unittest

import torch

from composable_model import ComposableFermiNet
from geometry import GeometricContext
from hamiltonian import MolecularSystem, compute_local_energy
from kinetic_modern import KineticEnergyCalculator, KineticMode
from model import FermiNet
from spin import SpinProjector


class TestPhysicalMatrix(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.coords = torch.zeros(1, 3, dtype=torch.float64)
        self.charges = torch.tensor([2.], dtype=torch.float64)

    def architectures(self):
        return [(FermiNet, {}), (ComposableFermiNet, {}),
                (ComposableFermiNet, {'interaction_type': 'psiformer'})]

    def model(self, cls, up, down, **kwargs):
        return cls(self.coords, self.charges, up, down, num_determinants=2,
                   hidden_dim_one=16, hidden_dim_two=4, num_layers=1,
                   use_cusp_jastrow=True, **kwargs).double()

    def test_hydrogen_exact_energy_and_zero_variance(self):
        class Hydrogen(torch.nn.Module):
            def forward_log_psi(self, r, geom_ctx=None):
                return -torch.linalg.vector_norm(r, dim=-1).sum(-1), r.new_ones(r.shape[0])
        directions = torch.cat((torch.eye(3), -torch.eye(3))).double()
        radii = torch.logspace(-6, .7, 12, dtype=torch.float64)
        points = (radii[:, None, None]*directions[None]).reshape(-1,1,3)
        system = MolecularSystem(self.coords, torch.ones(1,dtype=torch.float64), 1, 0)
        for mode in (KineticMode.REVERSE, KineticMode.FORWARD_JVP):
            for cached in (False, True):
                r = points.clone().requires_grad_()
                ctx = GeometricContext.compute(r, self.coords, eps=1e-6) if cached else None
                energy = compute_local_energy(Hydrogen(),system,r,geom_ctx=ctx,kinetic_mode=mode)[0]
                torch.testing.assert_close(energy, torch.full_like(energy,-.5), atol=2e-9, rtol=0)
                self.assertLess(energy.var().item(), 1e-18)

    def test_same_spin_antisymmetry_and_s_squared(self):
        for cls, kwargs in self.architectures():
            net = self.model(cls,2,2,spin_adapted=True,**kwargs)
            r = torch.randn(3,4,3,dtype=torch.float64)+.4
            with torch.no_grad():
                log, sign = net.forward_log_psi(r)
                log_swap, sign_swap = net.forward_log_psi(r[:,[1,0,2,3]])
                self.assertTrue((sign != 0).all())
                torch.testing.assert_close(log_swap,log,atol=1e-9,rtol=1e-9)
                torch.testing.assert_close(sign_swap,-sign,atol=0,rtol=0)
                scale = log.max()
                psi = sign*torch.exp(log-scale)
                residual = 2*psi
                for i in (0,1):
                    for j in (2,3):
                        perm = list(range(4)); perm[i],perm[j] = perm[j],perm[i]
                        lp,sp = net.forward_log_psi(r[:,perm])
                        residual -= sp*torch.exp(lp-scale)
                self.assertLess((residual.norm()/psi.norm()).item(),1e-9)

    def test_triplet_pair_limit_for_all_architectures(self):
        for cls, kwargs in self.architectures():
            net = self.model(cls,1,1,spin_projector=SpinProjector(1,1,'triplet'),**kwargs)
            system = MolecularSystem(self.coords,self.charges,1,1)
            energies = []
            for distance in (1e-3,1e-4,1e-5):
                r = torch.tensor([[[.5,.2,.3],[.5+distance,.2,.3]]],dtype=torch.float64,requires_grad=True)
                energies.append(compute_local_energy(net,system,r)[0].item())
            self.assertLess(abs(energies[-1]-energies[-2]),.1)

    def test_real_models_jvp_energy_and_parameter_derivative(self):
        for cls, kwargs in self.architectures():
            net = self.model(cls,1,1,spin_adapted=True,**kwargs)
            system = MolecularSystem(self.coords,self.charges,1,1)
            points = torch.randn(2,2,3,dtype=torch.float64)+.4
            results = []
            for mode in (KineticMode.REVERSE,KineticMode.FORWARD_JVP):
                energy = compute_local_energy(net,system,points.clone().requires_grad_(),
                                               kinetic_mode=mode,create_graph=True)[0]
                gradient = torch.autograd.grad(energy.sum(),net.det_weights)[0]
                results.append((energy.detach(),gradient.detach()))
            for got,expected in zip(results[1],results[0]):
                torch.testing.assert_close(got,expected,atol=1e-8,rtol=1e-7)

    def test_hutchinson_off_diagonal_variance(self):
        torch.manual_seed(101)
        a = torch.randn(6,6,dtype=torch.float64)
        a = a @ a.T + torch.eye(6,dtype=torch.float64)
        r = torch.randn(256,2,3,dtype=torch.float64,requires_grad=True)
        def wave(coords):
            x = coords.reshape(-1,6)
            return -.5*torch.einsum('bi,ij,bj->b',x,a,x), x.new_ones(x.shape[0])
        exact = -a.trace()
        estimates = []
        for probes in (1,64):
            lap = KineticEnergyCalculator.compute_kinetic_energy(wave,r,
                mode=KineticMode.HUTCHINSON,num_probes=probes)[1]
            error_std = (2*(a.square().sum()-a.diag().square().sum())/(256*probes)).sqrt()
            self.assertLess(abs(lap.mean().item()-exact.item()),5*error_std.item())
            estimates.append(lap.std().item())
        self.assertLess(estimates[1],.3*estimates[0])

    def test_conflicting_cusp_configuration_is_rejected(self):
        for cls, kwargs in self.architectures():
            with self.assertRaises(ValueError):
                self.model(cls,1,1,envelope_type='exponential',**kwargs)
            with self.assertRaises(ValueError):
                self.model(cls,1,1,smooth_cuspless=False,**kwargs)


if __name__ == '__main__':
    unittest.main()
