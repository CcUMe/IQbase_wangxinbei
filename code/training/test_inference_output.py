from pathlib import Path
import sys,json,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.inference_output import merge_prediction_shards


class InferenceOutputTests(unittest.TestCase):
    def test_shards_merge_in_original_order_and_preserve_public_fields(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);shards=[]
            for rank in [0,1]:
                path=root/f'{rank}.jsonl';shards.append(path)
                path.write_text(''.join(json.dumps({'row':i,'result':{'task':'modulation','sample_id':str(i),'prediction':'BPSK'}})+'\n'
                                        for i in range(rank,5,2)))
            out=root/'output.jsonl';self.assertEqual(merge_prediction_shards(shards,out),5)
            rows=[json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual([row['sample_id'] for row in rows],list(map(str,range(5))))
            self.assertTrue(all(set(row)=={'task','sample_id','prediction'} for row in rows))

    def test_invalid_shards_cannot_replace_existing_output(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'out.jsonl';out.write_text('original')
            a=root/'a';b=root/'b';a.write_text(json.dumps({'row':0,'result':{}})+'\n');b.write_text(a.read_text())
            with self.assertRaisesRegex(ValueError,'missing/duplicate'):merge_prediction_shards([a,b],out)
            self.assertEqual(out.read_text(),'original')

    def test_predictor_runs_without_gradients(self):
        import torch
        from infer import Predictor
        from models.rf_model import RFModel
        from utils.common import save_safetensors,write_json
        schema={'classification_labels':{'modulation':['a','b'],'protocol':['c','d']}}
        scaler={'mean':[0.]*7,'std':[1.]*7}
        model=RFModel({'width':32,'depth':1,'heads':4,'dropout':.2},schema,scaler,[[]]*7)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            save_safetensors(root/'model.safetensors',model.state_dict())
            write_json(root/'config.json',{'model_config':model.config,'schema':schema,'target_scaler':scaler,'parameter_grids':[[]]*7})
            predictor=Predictor(root,device='cpu',threads=2)
            flags=[]
            predictor.model.register_forward_hook(lambda m,a,o:flags.append((torch.is_grad_enabled(),o['modulation'].requires_grad)))
            out=predictor.predict([{'task':'modulation','sample_id':'0','iq':torch.randn(512,2).tolist()}])
            self.assertEqual(flags,[(False,False)])
            self.assertTrue(all(p.grad is None for p in predictor.model.parameters()))
            self.assertEqual(out[0]['sample_id'],'0')


if __name__=='__main__':unittest.main(verbosity=2)
