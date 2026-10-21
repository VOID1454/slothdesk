import unittest
from unittest.mock import patch
import app
class ReportTests(unittest.TestCase):
    def sample(self):
        return {k:(app.ROOT/'samples'/f'{k}.csv').read_text(encoding="utf-8") for k in ['pre','post','feedback']}
    def test_matched_cohort(self):
        r=app.analyse(self.sample())
        self.assertEqual((r['matched'],r['pre_mean'],r['post_mean'],r['gain_pp']),(4,57.5,80,22.5))
        self.assertEqual(r['pre_only'],['P005']);self.assertEqual(r['post_only'],['P006'])
    def test_invalid_scores(self):
        for score in ['NaN','inf','-1','11','hello']:
            with self.subTest(score=score),self.assertRaises(ValueError):
                app.assessment(f'participant_id,score,max_score\nA,{score},10','Pre')
    def test_duplicates(self):
        with self.assertRaisesRegex(ValueError,'duplicate'):
            app.assessment('participant_id,score,max_score\nA,5,10\nA,6,10','Pre')
    def test_no_overlap(self):
        d=self.sample();d['post']='participant_id,score,max_score\nB,5,10'
        with self.assertRaisesRegex(ValueError,'No matching'):app.analyse(d)
    def test_normalization(self):
        self.assertEqual(app.assessment('\ufeffparticipant_id,score,max_score\nA,7,20','Pre')['A'],35)
    def test_model_payload_and_no_identifiers(self):
        r=app.analyse(self.sample())
        with patch.object(app,'ollama',return_value={'message':{'content':'Strengths [F2]'}}) as model:
            self.assertEqual(app.summarize(r),'Strengths [F2]')
            payload=model.call_args.args[1]
            self.assertNotIn('P001',str(payload));self.assertFalse(payload['stream'])
    def test_no_feedback_and_size(self):
        with self.assertRaises(ValueError):app.summarize({'feedback':[]})
        with self.assertRaises(ValueError):app.summarize({'feedback':[{'source':'F2','text':'x'*18001}]})
if __name__=='__main__':unittest.main()
