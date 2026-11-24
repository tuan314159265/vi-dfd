"""Batch-one CPU latency on fixed input; record fixed-horizon decode convention."""
import argparse
import json
import platform
import time
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--warmup',type=int,default=50)
    p.add_argument('--runs',type=int,default=500)
    args=p.parse_args()
    if args.runs<1 or args.warmup<0: p.error('Invalid timing counts')
    import numpy as np
    import onnxruntime as ort
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    session=ort.InferenceSession(str(args.model),sess_options=options,providers=['CPUExecutionProvider'])
    x=np.random.default_rng(42).uniform(-1,1,(1,3,32,128)).astype('float32')
    for _ in range(args.warmup):session.run(None,{'images':x})
    elapsed=[]
    for _ in range(args.runs):
        start=time.perf_counter();session.run(None,{'images':x});elapsed.append((time.perf_counter()-start)*1000)
    result={'model':str(args.model),'median_ms':statistics.median(elapsed),'min_ms':min(elapsed),
            'warmup':args.warmup,'runs':args.runs,'threads':4,'batch_size':1,'onnxruntime':ort.__version__,
            'platform':platform.platform(),'cpu':platform.processor(),
            'decode':'fixed max horizon AR+refinement; excludes preprocessing and string decoding',
            'timings_ms':elapsed}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('median_ms:',result['median_ms'])
    return 0

if __name__=='__main__':
    raise SystemExit(main())
