import csv, unittest
from src.submission import write_submission

class SubmissionTests(unittest.TestCase):
    def test_all_formats(self):
        cases={
          'KIS':([{'video_id':'L01_V001','frame_id':10}],['video_id','frame_id']),
          'QA':([{'video_id':'L01_V001','frame_id':10,'answer':'blue'}],['video_id','frame_id','answer']),
          'TRAKE':([{'video_id':'L01_V001','hits':[{'frame_idx':10},{'frame_idx':20}]}],['video_id','frame_id_1','frame_id_2'])}
        for kind,(rows,header) in cases.items():
            with self.subTest(kind=kind):
                with open(write_submission(kind,rows),encoding='utf-8-sig') as f: self.assertEqual(next(csv.reader(f)),header)

if __name__=='__main__': unittest.main()
