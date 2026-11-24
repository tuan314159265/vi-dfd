"""Export full PARSeq AR+refinement at batch 1 and fixed maximum horizon."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--verify-data', type=Path, required=True, help='validation LMDB directory')
    p.add_argument('--verify-samples', type=int, default=32)
    args = p.parse_args()
    import numpy as np
    import torch
    import onnx
    import onnxruntime as ort
    from dfd.parseq import activate_upstream
    activate_upstream()
    from dfd.evaluation import load_parseq_checkpoint, read_lmdb
    from dfd.vietchar import DecompositionTable
    import hashlib
    def digest(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as handle:
            for block in iter(lambda: handle.read(1 << 20), b''):
                h.update(block)
        return h.hexdigest()
    from strhub.data.module import SceneTextDataModule
    from PIL import Image
    import io
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    system = load_parseq_checkpoint(str(args.checkpoint), DecompositionTable.from_characters(), 'cpu')
    class FullRecognizer(torch.nn.Module):
        def __init__(self, system):
            super().__init__()
            self.model = system.model
            self.tokenizer = system.tokenizer
        def forward(self, images):
            # Explicit max_length disables data-dependent Python early exit during tracing.
            return self.model(self.tokenizer, images, max_length=self.model.max_label_length)
    wrapper = FullRecognizer(system).eval()
    samples = read_lmdb(args.verify_data)[:args.verify_samples]
    if not samples:
        raise ValueError('Verification samples required')
    transform = SceneTextDataModule.get_transform(tuple(system.hparams.img_size))
    def tensor(sample):
        return transform(Image.open(io.BytesIO(sample.image)).convert('RGB')).unsqueeze(0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(wrapper, tensor(samples[0]), str(args.output), opset_version=17,
                          input_names=['images'], output_names=['logits'], do_constant_folding=True)
    graph = onnx.shape_inference.infer_shapes(onnx.load(str(args.output)))
    types = {v.name: v.type.tensor_type.elem_type for v in graph.graph.value_info}
    nodes = []
    for node in graph.graph.node:
        if node.op_type == 'Where' and types.get(node.output[0]) == onnx.TensorProto.BOOL:
            cond, yes, no = node.input
            prefix = node.output[0] + '_bool_select'
            nodes.extend([
                onnx.helper.make_node('Not', [cond], [prefix + '_not']),
                onnx.helper.make_node('And', [cond, yes], [prefix + '_yes']),
                onnx.helper.make_node('And', [prefix + '_not', no], [prefix + '_no']),
                onnx.helper.make_node('Or', [prefix + '_yes', prefix + '_no'], list(node.output)),
            ])
        else:
            nodes.append(node)
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    onnx.save(graph, str(args.output))
    onnx.checker.check_model(str(args.output))
    options = ort.SessionOptions(); options.intra_op_num_threads = 4; options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.output), sess_options=options, providers=['CPUExecutionProvider'])
    maximum_error = 0.
    with torch.inference_mode():
        for sample in samples:
            x = tensor(sample)
            original, _ = system.tokenizer.decode(system(x).softmax(-1))
            expected = wrapper(x).numpy()
            actual = session.run(None, {'images': x.numpy()})[0]
            np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=2e-3)
            decoded, _ = system.tokenizer.decode(torch.from_numpy(actual).softmax(-1))
            if decoded != original:
                raise ValueError(f'Decoded mismatch: {decoded} vs {original}; export is not equivalent')
            maximum_error = max(maximum_error, float(np.max(np.abs(actual - expected))))
    record = {'checkpoint_sha256': digest(args.checkpoint), 'onnx_sha256': digest(args.output),
              'characters': list(system.tokenizer._itos)[:len(DecompositionTable.from_characters())],
              'max_label_length': system.model.max_label_length, 'decode_ar': system.model.decode_ar,
              'refine_iters': system.model.refine_iters, 'batch_size': 1, 'input_shape': [1,3,32,128],
              'horizon': 'fixed maximum; no early termination', 'verified_samples': len(samples),
              'max_abs_error': maximum_error, 'precision': 'fp32'}
    args.output.with_suffix('.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
