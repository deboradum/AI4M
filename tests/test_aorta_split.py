import unittest

import numpy as np
from scipy.ndimage import distance_transform_edt

from retrieve_aorta import split_once


def synthetic_class1() -> np.ndarray:
    # A folded class 1 shaped like the real patients: a thick aorta cylinder, a thin
    # esophagus tube beside it, and a bridge merging the two for a few slices (real
    # class 1 is one blob, the organs touching in most slices).
    #
    # The aorta cross-section is 615 mm^2, above the 300 mm^2 seed threshold, and the
    # tube is about 13 mm^2 - the separation the rule relies on (real aortas run 400
    # to 1400 mm^2). The tube is broken in z with the merged slices between the two
    # stretches: those slices hold a single component, so they seed nothing, which
    # leaves the esophagus seeds as two separate components in z - the situation that
    # makes the seeding rule observable.
    volume = np.zeros((64, 64, 30), dtype=bool)
    xs, ys = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    aorta = (xs - 24) ** 2 + (ys - 32) ** 2 <= 14 ** 2
    tube = (xs - 48) ** 2 + (ys - 32) ** 2 <= 2 ** 2

    for z in range(5, 26):
        volume[:, :, z] |= aorta
    for z in list(range(5, 13)) + list(range(16, 26)):
        volume[:, :, z] |= tube
    for z in (13, 14, 15):                      # bridge: class 1 is one component here
        volume[36:49, 31:34, z] = True
    return volume


class TestSplitOnce(unittest.TestCase):
    def setUp(self) -> None:
        self.spacing = (1.0, 1.0, 1.0)
        self.folded = synthetic_class1()
        self.dt = distance_transform_edt(self.folded, sampling=self.spacing)
        self.aorta, self.eso = split_once(self.folded, self.dt, 300.0, self.spacing)

    def test_every_esophagus_seed_component_is_recovered(self) -> None:
        # The bug: seeding the esophagus from only the largest component left the other
        # stretch of tube with no marker at all, and every class-1 voxel without a
        # basin (aorta = folded & ~esophagus) silently became aorta. On Patient_15 that
        # cost 37 of the 113 slices class 1 occupies, so the esophagus stopped at
        # z=62 halfway down the thorax.
        for z in list(range(5, 13)) + list(range(16, 26)):
            self.assertTrue(self.eso[:, :, z].any(),
                            f"esophagus seed stretch missing at z={z}")

    def test_tube_below_the_break_is_esophagus_not_aorta(self) -> None:
        # The tube centre, in the stretch that used to be swallowed by the aorta.
        self.assertTrue(self.eso[48, 32, 8], "tube at z=8 must be esophagus")
        self.assertFalse(self.aorta[48, 32, 8], "tube at z=8 must not be aorta")

    def test_split_stays_complete_and_disjoint(self) -> None:
        self.assertFalse((self.aorta & self.eso).any(), "organs must not overlap")
        self.assertTrue(np.array_equal(self.aorta | self.eso, self.folded),
                        "every class-1 voxel must land in exactly one organ")

    def test_aorta_keeps_the_thick_cylinder(self) -> None:
        self.assertTrue(self.aorta[24, 32, 8], "aorta core must stay aorta")
        self.assertFalse(self.eso[24, 32, 8], "aorta core must not be esophagus")


if __name__ == "__main__":
    unittest.main()