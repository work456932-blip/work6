"""Analytical checks only; these toy arrays are not benchmark observations."""
import unittest
import numpy as np
from PIL import Image
from gsdvq.config import Config
from gsdvq.support import count_bin,empirical_pmf
from gsdvq.numerics import gsd_statistics,crps_point,wasserstein
from gsdvq.masking import FixedKPlan,fixed_k_mask,components,remove_targets,boundary_perturbation
from gsdvq.metrics import expected_count_report,rating_report

def delta(n):
    p=np.zeros(102)
    p[count_bin(n)]=1.
    return p

class EquationsAndMasks(unittest.TestCase):
    def test_dominance_direction(self):
        result=gsd_statistics(delta(10),delta(2))
        self.assertEqual(float(result["s_dom"]),1.)
        self.assertEqual(float(result["a_inc"]),0.)
        inverse=gsd_statistics(delta(2),delta(10))
        self.assertEqual(float(inverse["s_dom"]),-1.)

    def test_crossing_has_incomparability(self):
        result=gsd_statistics(.5*delta(0)+.5*delta(10),delta(5))
        self.assertEqual(float(result["s_dom"]),0.)
        self.assertEqual(float(result["a_inc"]),.5)

    def test_overflow_is_distinct(self):
        self.assertEqual(count_bin(100),100)
        self.assertEqual(count_bin(101),101)
        self.assertEqual(count_bin(250),101)
        p=empirical_pmf([100,150])
        self.assertEqual(p[100],.5)
        self.assertEqual(p[101],.5)

    def test_count_scores(self):
        self.assertEqual(crps_point(delta(2),5),3.)
        self.assertEqual(float(wasserstein(delta(2),delta(5))),3.)
        self.assertEqual(crps_point(delta(5),5),0.)

    def test_human_distribution_validation(self):
        report=expected_count_report([{"id":"x","pexp":delta(5).tolist()}],[{"id":"x","expected_counts":[5]*10}])
        self.assertEqual(report["ncrps"],0.)
        self.assertEqual(report["nw1"],0.)

    def test_fixed_k_preserves_targets(self):
        target=np.zeros((64,64),bool)
        target[4:10,4:10]=True
        plan=FixedKPlan(3,[.003],[1.])
        mask=fixed_k_mask(target,np.zeros_like(target),plan,np.random.default_rng(0))
        self.assertEqual(components(mask),3)
        self.assertTrue(np.all(mask[target]))
        enlarged=boundary_perturbation(target,np.random.default_rng(1))
        self.assertTrue(np.all(enlarged[target]))

    def test_mask_outside_pixels_are_preserved(self):
        rgb=np.arange(12*12*3,dtype=np.uint8).reshape(12,12,3)
        mask=np.zeros((12,12),bool)
        mask[3:7,3:7]=True
        result=np.asarray(remove_targets(Image.fromarray(rgb),mask,"gray"))
        np.testing.assert_array_equal(result[~mask],rgb[~mask])
        self.assertTrue((result[mask]==127).all())

    def test_inpaint_requires_real_adapter(self):
        with self.assertRaises(ValueError):
            remove_targets(Image.new("RGB",(4,4)),np.ones((4,4),bool),"inpaint")

if __name__=="__main__":
    unittest.main()
