import sys
import unittest
from collections import Counter
from pathlib import Path

from videovera.data.displacement.generate import slots_for, plan_label, path_key, select_batch, variants, balanced_letters
from videovera.data.displacement.geometry import DIRECTIONS


class BalancedTrainingTests(unittest.TestCase):
    def test_option_letters_balanced_inside_each_reference_direction_cell(self):
        slots = slots_for(1000)
        keys = balanced_letters(slots)
        self.assertEqual(Counter(keys.values()),dict.fromkeys('ABCD',250))
        for ref in ('start','end'):
            self.assertEqual(Counter(keys[s['index']] for s in slots if s['reference'] == ref),
                             dict.fromkeys('ABCD',125))
            for direction in DIRECTIONS:
                counts = Counter(keys[s['index']] for s in slots if s['reference'] == ref and s['direction'] == direction)
                self.assertEqual(set(counts),set('ABCD'))
                self.assertLessEqual(max(counts.values())-min(counts.values()),1)

    def test_1000_joint_quotas(self):
        slots = slots_for(1000)
        self.assertEqual(Counter(s['reference'] for s in slots), {'start':500, 'end':500})
        self.assertEqual(Counter(s['letter'] for s in slots), dict.fromkeys('ABCD',250))
        counts = Counter(s['direction'] for s in slots)
        self.assertEqual(counts['same_location'],112)
        self.assertTrue(all(counts[d] == 111 for d in DIRECTIONS if d != 'same_location'))
        self.assertEqual(sum(s['required_mode'] == 'straight' for s in slots),50)
        for direction in DIRECTIONS:
            by_ref = Counter(s['reference'] for s in slots if s['direction'] == direction)
            self.assertLessEqual(abs(by_ref['start']-by_ref['end']),1)

    def test_reference_orientation_not_reversed_answer(self):
        points = [[0,0,0],[0,0,-2]]
        self.assertEqual(plan_label(points,0,0,'start'),'front')
        self.assertEqual(plan_label(points,0,180,'end'),'front')
        self.assertEqual(plan_label(points,0,90,'end'),'left')
        for ref in ('start','end'):
            self.assertEqual(plan_label(points+[points[0]],0,180,ref),'same_location')

    def test_geometric_parent_dedup_ignores_traversal_direction(self):
        points = [[0,0,0],[0,0,-2],[2,0,-2]]
        self.assertEqual(path_key('scene',points),path_key('scene',points[::-1]))
        self.assertNotEqual(path_key('scene',points),path_key('other',points))

    def test_return_variants_follow_source_legs_back(self):
        points = [[0,0,0],[0,0,-2],[2,0,-2]]
        payload = {'navmesh_route':{'simplified_agent_points_xyz_m':points},
                   'camera_trajectory':{'frames':[{'rotation_xyzw':[0,0,0,1]}]*2}}
        pool = {mode:(p,a,b) for mode,p,a,b in variants(payload)}
        self.assertEqual(pool['return_two_legs'][0],points+[points[1],points[0]])
        self.assertEqual(pool['straight'][0],points[:2])

    def test_selector_will_not_pad_by_reusing_routes(self):
        slot = {'index':0,'reference':'start','direction':'front','required_mode':None}
        c = {'candidate_id':'c','reference':'start','direction':'front','parent_key':'p',
             'route_fingerprint':'f','scene':'scene','mode':'original'}
        self.assertEqual(len(select_batch([slot],[c],{},set(),42)),1)
        with self.assertRaisesRegex(ValueError,'insufficient_unique_routes'):
            select_batch([slot,{**slot,'index':1}],[c],{},set(),42)
        with self.assertRaisesRegex(ValueError,'insufficient_unique_routes'):
            select_batch([slot],[c,{**c,'candidate_id':'duplicate_annotation'}],{}, {'c'},42)


if __name__ == '__main__':
    unittest.main()
