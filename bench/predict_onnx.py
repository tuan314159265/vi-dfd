"""Write source_id,prediction CSVs for the shared eval.py scorer (batch 1)."""
import argparse
import csv
import io
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def decode_ids(ids, characters):
    output=[]
    for token in ids:
        if int(token)==0:
            break
        output.append(characters[int(token)])
    return ''.join(output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True,help='Root containing lmdb/')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--splits',nargs='+',default=['test','unseen_test'],choices=['validation','test','unseen_test'])
    args=p.parse_args()
    import numpy as np
    import onnxruntime as ort
    from PIL import Image
    from dfd.evaluation import read_lmdb, split_dir
    from dfd.teacher_data import image_transform
    meta=json.loads(args.model.with_suffix('.json').read_text())
    opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
    session=ort.InferenceSession(str(args.model),sess_options=opts,providers=['CPUExecutionProvider'])
    transform=image_transform(False)
    args.output.mkdir(parents=True,exist_ok=True)
    for split in args.splits:
        samples=read_lmdb(args.data_root/'lmdb'/split_dir(split))
        with (args.output/f'{split}.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.writer(f);w.writerow(['source_id','prediction'])
            for sample in samples:
                image=Image.open(io.BytesIO(sample.image)).convert('RGB')
                logits=session.run(None,{'images':transform(image).unsqueeze(0).numpy()})[0]
                text=decode_ids(np.argmax(logits[0],axis=-1),meta['characters'])
                w.writerow([sample.source_id,text])
        print(split,len(samples),'predictions written',flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
