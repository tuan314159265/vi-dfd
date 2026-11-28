"""Dynamic INT8 MatMul/Gemm weight quantization; leave unsupported Conv in FP32."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    from onnxruntime.quantization import quantize_dynamic, QuantType
    import onnxruntime as ort
    import onnx
    args.output.parent.mkdir(parents=True,exist_ok=True)
    quantize_dynamic(str(args.input),str(args.output),weight_type=QuantType.QInt8,
                     op_types_to_quantize=['MatMul','Gemm'],per_channel=True,
                     extra_options={'MatMulConstBOnly':True})
    onnx.checker.check_model(str(args.output))
    opts=ort.SessionOptions();opts.intra_op_num_threads=4
    ort.InferenceSession(str(args.output),sess_options=opts,providers=['CPUExecutionProvider'])
    meta=json.loads(args.input.with_suffix('.json').read_text())
    meta.update(precision='int8_dynamic',quantized_ops=['MatMul','Gemm'],
                note='Mixed precision: unsupported operators, including Conv, remain FP32. Accuracy must be scored separately.')
    # The original hash describes the source FP32 graph, not this quantized graph.
    meta['source_fp32_sha256']=meta.pop('onnx_sha256')
    import hashlib
    meta['onnx_sha256']=hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('Quantized:',args.output)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
