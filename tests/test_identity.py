import unittest
from src.identity import pick, stable_pk

class IdentityTests(unittest.TestCase):
    def test_map_columns(self):
        row={'n':'10','pts_time':'10.0','fps':'30','frame_idx':'300'}
        self.assertEqual(int(pick(row,'n')),10)
        self.assertEqual(int(pick(row,'frame_idx')),300)

    def test_stable_primary_key(self):
        self.assertEqual(stable_pk('L21_V001',10),21_000_010_000_010)

if __name__=='__main__': unittest.main()
