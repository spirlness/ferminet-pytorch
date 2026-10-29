"""Independent mathematical checks for the fixed-wavefunction diagnosis."""

import unittest

import torch
from torch import nn

from ferminet_pytorch.loss import clip_local_energy
from scripts.diagnose_frozen_tails import (
    cross_gradient_statistics,
    gradient_vectors,
    off_diagonal_dot,
    summarize_geometry,
)
from scripts.analyze_frozen_tail_study import bias_statistics
from scripts.attribute_frozen_gradient_bias import region_gradient_vectors


class LinearLogWavefunction(nn.Module):
    def __init__(self,shift=0.):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(.2, dtype=torch.float64))
        self.shift=shift

    def forward_log_psi(self, coordinates):
        score = coordinates[:, 0]
        return self.theta * (score+self.shift), torch.ones_like(score)


class TestFrozenDiagnosis(unittest.TestCase):
    def test_gradient_matches_sample_covariance_and_training_rule(self):
        model = LinearLogWavefunction()
        positions = torch.tensor([[-2.], [-.5], [1.], [3.]], dtype=torch.float64)
        energies = torch.tensor([-2., -1.2, -1., 4.], dtype=torch.float64)
        vectors = gradient_vectors(model, positions, energies)
        covariance = torch.cov(torch.stack((energies, positions[:, 0])))[0, 1]
        self.assertAlmostEqual(float(vectors[0, 0]), float(2*covariance), places=12)
        clipped, _, _ = clip_local_energy(energies)
        expected = 2*((clipped-clipped.mean())*positions[:, 0]).mean()
        self.assertAlmostEqual(float(vectors[1, 0]), float(expected), places=12)
        self.assertEqual(float(model.theta.detach()), .2)

    def test_cross_replica_noise_is_not_mistaken_for_signal(self):
        noise = torch.tensor([[1.,0.], [-1.,0.], [0.,1.], [0.,-1.]], dtype=torch.float64)
        self.assertLess(off_diagonal_dot(noise, noise), 0)
        result = cross_gradient_statistics(torch.stack((noise,noise,noise),dim=1))
        self.assertEqual(result['rules']['current']['verdict'], 'inconclusive')

    def test_independent_descent_and_ascent_are_distinguished(self):
        raw = torch.tensor([[1.,0.]]*4, dtype=torch.float64)
        ascent = -raw
        result = cross_gradient_statistics(torch.stack((raw,raw,ascent),dim=1))
        self.assertEqual(result['rules']['current']['verdict'], 'descent_supported')
        self.assertEqual(result['rules']['paper_hard']['verdict'], 'ascent_supported')

    def test_spatial_flags_preserve_overlap_and_sample_count(self):
        data = {
            'energy':torch.tensor([[[-4.,-1.,-1.,-1.]]],dtype=torch.float64),
            'min_en':torch.tensor([[[.01,1.,1.,1.]]]),
            'max_nearest_nucleus':torch.tensor([[[4.,1.,1.,1.]]]),
            'ee':torch.tensor([[[.01,1.,1.,1.]]]),
            'cancellation':torch.ones(1,1,4),
            'sign':torch.ones(1,1,4),
        }
        result = summarize_geometry(data)
        self.assertEqual(result['samples'],4)
        self.assertEqual(result['extreme_count'],1)
        self.assertTrue(result['regions_overlap'])
        self.assertEqual(result['regions']['near_nucleus_lt_0.05']['extreme_count'],1)
        self.assertEqual(result['regions']['far_nearest_nucleus_gt_3']['extreme_count'],1)
        self.assertEqual(sum(row['count'] for row in result['far_distance_bins']),4)

    def test_systematic_bias_uses_cross_replica_squared_signal(self):
        raw = torch.tensor([1.,0.], dtype=torch.float64)
        clipped = torch.tensor([0.,1.], dtype=torch.float64)
        block = torch.stack((raw,clipped,raw))
        result = bias_statistics(block.expand(4,2,3,2).clone())
        signal = result['current']['raw_minus_clipped_squared_signal']
        self.assertEqual(signal['mean'],2.)
        self.assertEqual(signal['ci95_approx'],[2.,2.])

    def test_region_attribution_is_invariant_to_wavefunction_normalization(self):
        positions=torch.tensor([[-2.],[-.5],[1.],[3.]],dtype=torch.float64)
        energy=torch.tensor([-2.,-1.2,-1.,4.],dtype=torch.float64)
        masks=(torch.tensor([True,False,False,False]),torch.tensor([False,True,True,True]))
        a=region_gradient_vectors(LinearLogWavefunction(),positions,energy,masks)
        b=region_gradient_vectors(LinearLogWavefunction(100.),positions,energy,masks)
        torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
        all_gradients=gradient_vectors(LinearLogWavefunction(),positions,energy)
        torch.testing.assert_close(a.sum(0),all_gradients[0]-all_gradients[1])
        with_empty=region_gradient_vectors(LinearLogWavefunction(),positions,energy,
            masks+(torch.zeros(4,dtype=torch.bool),))
        torch.testing.assert_close(with_empty[:2],a)
        self.assertEqual(float(with_empty[-1].abs().sum()),0.)


if __name__=='__main__':
    unittest.main()
