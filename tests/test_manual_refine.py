import unittest
import sys
import types
from types import SimpleNamespace

# The service is tested with fakes and never opens Milvus or loads ML models.
fake_search=types.ModuleType('src.search_engine')
fake_search.SearchEngine=object
fake_images=types.ModuleType('src.image_cache')
fake_images.get_image=lambda *args,**kwargs: None
fake_vqa=types.ModuleType('src.visual_qa')
fake_vqa.VisualQA=object
fake_settings=types.ModuleType('src.settings')
fake_settings.settings=SimpleNamespace(auto_frame_refine=False,refine_top_n=0)
sys.modules.setdefault('src.search_engine',fake_search)
sys.modules.setdefault('src.image_cache',fake_images)
sys.modules.setdefault('src.visual_qa',fake_vqa)
sys.modules.setdefault('src.settings',fake_settings)

from frontend.retrieval_service import RetrievalService


class FakeTranslator:
    def to_english(self, text):
        return text


class FakeRefiner:
    def refine(self, video_id, coarse_frame, fps, query):
        return {
            'frame_idx': coarse_frame + 2,
            'refined': True,
            'refined_image_path': f'{video_id}_{coarse_frame + 2}.jpg',
        }


class FakeVQA:
    def answer(self, image_path, question):
        return 'red'


def make_service():
    service=RetrievalService.__new__(RetrievalService)
    service.engine=SimpleNamespace(
        translator=FakeTranslator(),
        refiner=FakeRefiner(),
    )
    service.vqa=FakeVQA()
    return service


class ManualRefineTests(unittest.TestCase):
    def test_qa_extracts_location_from_ocr_context(self):
        text=('TRÁI TIM TỪ HÀ NỘI ĐƯỢC VẬN CHUYỂN CẤP TỐC '
              'VÊ HUẾ GHÉP CHO BỆNH NHÂN')
        answer=RetrievalService._answer_from_ocr(
            'Trái tim được vận chuyển đến đâu?',text
        )
        self.assertEqual(answer,'HUẾ')

    def test_promote_result_moves_selected_row_and_resets_ranks(self):
        rows=[
            {'rank':1,'video_id':'L01_V001','frame_idx':10},
            {'rank':2,'video_id':'L01_V002','frame_idx':20},
            {'rank':3,'video_id':'L01_V003','frame_idx':30},
        ]
        updated,_=make_service().promote_result(rows,3)
        self.assertEqual([row['video_id'] for row in updated],
                         ['L01_V003','L01_V001','L01_V002'])
        self.assertEqual([row['rank'] for row in updated],[1,2,3])
        self.assertEqual([row['video_id'] for row in rows],
                         ['L01_V001','L01_V002','L01_V003'])

    def test_delete_result_removes_selected_row_and_resets_ranks(self):
        rows=[
            {'rank':1,'video_id':'L01_V001','frame_idx':10},
            {'rank':2,'video_id':'L01_V002','frame_idx':20},
            {'rank':3,'video_id':'L01_V003','frame_idx':30},
        ]
        updated,_=make_service().delete_result(rows,2)
        self.assertEqual([row['video_id'] for row in updated],['L01_V001','L01_V003'])
        self.assertEqual([row['rank'] for row in updated],[1,2])

    def test_kis_refines_only_selected_rank(self):
        rows=[
            {'video_id':'L01_V001','frame_idx':10,'frame_id':10,'fps':25,'image_path':'a.jpg'},
            {'video_id':'L01_V002','frame_idx':20,'frame_id':20,'fps':25,'image_path':'b.jpg'},
        ]
        updated,_=make_service().refine_kis_result(rows,2,'query')
        self.assertEqual(updated[0]['frame_idx'],10)
        self.assertEqual(updated[1]['frame_idx'],22)
        self.assertEqual(updated[1]['frame_id'],22)
        self.assertEqual(rows[1]['frame_idx'],20)

    def test_qa_reruns_answer_on_refined_image(self):
        rows=[{'video_id':'L01_V001','frame_idx':10,'frame_id':10,'fps':25,
               'image_path':'a.jpg','answer':'blue'}]
        updated,_=make_service().refine_qa_result(rows,1,'event','color?')
        self.assertEqual(updated[0]['frame_idx'],12)
        self.assertEqual(updated[0]['answer'],'red')

    def test_trake_refines_all_events_in_order(self):
        rows=[{'video_id':'L01_V001','hits':[
            {'frame_idx':10,'fps':25},{'frame_idx':20,'fps':25},{'frame_idx':30,'fps':25}
        ]}]
        updated,_=make_service().refine_trake_result(rows,1,['a','b','c'])
        self.assertEqual([x['frame_idx'] for x in updated[0]['hits']],[12,22,32])


if __name__=='__main__':
    unittest.main()
