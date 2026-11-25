"""E1 CRNN-CTC and upstream VietOCR adapters, with a shared VinText training loop."""
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from dfd.runtime import prepare_run, seed_everything
from dfd.vietchar import DecompositionTable

VIETOCR_COMMIT = 'fe8c3a7fc714aec57ab81cec844eb3adf0c1636c'
VIETOCR_WEIGHTS = 'https://vocr.vn/data/vietocr/vgg_seq2seq.pth'
VIETOCR_WEIGHTS_SHA256 = '0921503a41375a0584268e23ef3d414ea478a8fe8777865c7745d38f2d0bc5db'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_vietocr_weights(path):
    actual = digest(path)
    if actual != VIETOCR_WEIGHTS_SHA256:
        raise ValueError(f'VietOCR weights SHA-256 mismatch for {path}: '
                         f'expected {VIETOCR_WEIGHTS_SHA256}, found {actual}')
    return actual


class CRNN(nn.Module):
    """RGB CNN -> two bidirectional LSTM layers -> CTC, blank token 0.

    Local baseline architecture, not a reproduction of a named pretrained checkpoint.
    Width 128 becomes 64 time steps, sufficient for the training limit of 25
    except repeated-character CTC alignments (checked explicitly by the trainer).
    """
    def __init__(self, characters):
        super().__init__()
        self.characters = characters
        layers = []
        channels = [3, 64, 128, 256, 256, 512]
        for i, (a, b) in enumerate(zip(channels, channels[1:])):
            layers += [nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU()]
            if i == 0:
                layers += [nn.MaxPool2d(2)]
            elif i < 4:
                layers += [nn.MaxPool2d((2, 1))]
        self.cnn = nn.Sequential(*layers)
        self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True, batch_first=True, dropout=.1)
        self.head = nn.Linear(512, len(characters) + 1)

    def forward(self, images):
        features = self.cnn(images).mean(dim=2).transpose(1, 2)
        return self.head(self.rnn(features)[0])

    def loss(self, images, labels):
        logits = self(images)
        ids = {c: i + 1 for i, c in enumerate(self.characters)}
        sequences = [[ids[c] for c in text] for text in labels]
        # A repeated neighbour requires a blank between the two targets.
        if any(len(seq) + sum(a == b for a, b in zip(seq, seq[1:])) > logits.shape[1] for seq in sequences):
            raise ValueError('CRNN CTC alignment is impossible at width 128 for a training label')
        targets = torch.tensor([i for seq in sequences for i in seq], device=images.device)
        lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
        inputs = torch.full((len(labels),), logits.shape[1], dtype=torch.long)
        return nn.functional.ctc_loss(logits.float().log_softmax(-1).transpose(0, 1), targets,
                                      inputs, lengths, blank=0, zero_infinity=False)

    def predict(self, images):
        return [ctc_decode(row, self.characters) for row in self(images).argmax(-1).tolist()]


def ctc_decode(tokens, characters):
    output, previous = [], None
    for token in tokens:
        if token and token != previous:
            output.append(characters[token - 1])
        previous = token
    return ''.join(output)


def activate_vietocr():
    root = Path(os.environ.get('VIETOCR_ROOT', Path.home() / '.cache/vi-dfd/vietocr')).expanduser().resolve()
    if not (root / 'vietocr/model/transformerocr.py').is_file():
        raise FileNotFoundError(f'Clone https://github.com/pbcquoc/vietocr into {root}, checkout {VIETOCR_COMMIT}')
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != VIETOCR_COMMIT:
        raise ValueError(f'VietOCR must use commit {VIETOCR_COMMIT}, found {revision}')
    if subprocess.check_output(['git', '-C', str(root), 'diff', 'HEAD', '--name-only'], text=True).strip():
        raise ValueError('VietOCR upstream has uncommitted modifications')
    sys.path.insert(0, str(root))
    return root


class VietOCRAdapter(nn.Module):
    def __init__(self, pretrained=True, checkpoint=None):
        super().__init__()
        root = activate_vietocr()
        from vietocr.model.transformerocr import VietOCR
        from vietocr.model.vocab import Vocab
        base = yaml.safe_load((root / 'config/base.yml').read_text())
        conf = yaml.safe_load((root / 'config/vgg-seq2seq.yml').read_text())
        self.vocab = Vocab(base['vocab'])
        self.characters = base['vocab']
        # Full OCR checkpoint already contains CNN weights; avoid a second ImageNet download.
        self.model = VietOCR(len(self.vocab), conf['backbone'], {**conf['cnn'], 'pretrained': False},
                             conf['transformer'], conf['seq_modeling'])
        self.initialization = {'upstream_commit': VIETOCR_COMMIT, 'pretrained': pretrained}
        if pretrained:
            if checkpoint:
                path = Path(checkpoint).expanduser()
            else:
                path = Path(torch.hub.get_dir()) / 'checkpoints' / VIETOCR_WEIGHTS.rsplit('/', 1)[-1]
                if not path.is_file():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.hub.download_url_to_file(VIETOCR_WEIGHTS, str(path))
            weights_sha256 = verify_vietocr_weights(path)
            state = torch.load(path, map_location='cpu', weights_only=True)
            self.model.load_state_dict(state, strict=True)
            self.initialization.update(weights_sha256=weights_sha256, weights_source=str(path))

    def loss(self, images, labels):
        sequences = [self.vocab.encode(text) for text in labels]
        width = max(map(len, sequences))
        tokens = torch.full((len(labels), width), self.vocab.pad, device=images.device, dtype=torch.long)
        for i, seq in enumerate(sequences):
            tokens[i, :len(seq)] = torch.tensor(seq, device=images.device)
        logits = self.model((images + 1) / 2, tokens[:, :-1].T, None)
        return nn.functional.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].reshape(-1), ignore_index=self.vocab.pad)

    def predict(self, images):
        # Greedy decoding using upstream recurrent encoder/decoder APIs, bounded length.
        memory = self.model.transformer.forward_encoder(self.model.cnn((images + 1) / 2))
        tokens = torch.full((1, len(images)), self.vocab.go, device=images.device, dtype=torch.long)
        ended = torch.zeros(len(images), device=images.device, dtype=torch.bool)
        for _ in range(26):  # up to 25 training characters plus EOS
            logits, memory = self.model.transformer.forward_decoder(tokens, memory)
            next_token = logits[:, -1].argmax(-1)
            next_token = torch.where(ended, self.vocab.eos, next_token)
            tokens = torch.cat([tokens, next_token[None]], dim=0)
            ended |= next_token == self.vocab.eos
            if ended.all():
                break
        result = []
        for row in tokens[1:].T.tolist():
            end = row.index(self.vocab.eos) if self.vocab.eos in row else len(row)
            # Preserve reserved-token errors instead of silently deleting them.
            result.append(''.join(self.vocab.i2c[t] if t >= 4 else '\ufffd' for t in row[:end]))
        return result


def build_baseline(config, pretrained=None):
    chars = ''.join(DecompositionTable.from_characters().characters[1:])
    if config['variant'] == 'crnn':
        if config['model'].get('pretrained'):
            raise ValueError('CRNN has no pretrained checkpoint; set model.pretrained: false')
        model = CRNN(chars)
        model.initialization = {'pretrained': False, 'architecture': 'local CNN-2xBiLSTM256-CTC'}
        return model
    if config['variant'] == 'vietocr':
        return VietOCRAdapter(config['model']['pretrained'] if pretrained is None else pretrained,
                              config['model'].get('pretrained_checkpoint'))
    raise ValueError('Expected crnn or vietocr')


@torch.inference_mode()
def predict_loader(model, loader, device):
    model.eval()
    rows = []
    for images, labels, ids in loader:
        predictions = model.predict(images.to(device))
        rows.extend(zip(ids, labels, predictions))
    return rows


def validation_accuracy(rows):
    from dfd.metrics import normalize_text
    table = DecompositionTable.from_characters()
    supported = set(table.characters[1:])
    return sum(set(label).issubset(supported) and normalize_text(label).casefold() == normalize_text(pred).casefold()
               for _, label, pred in rows) / len(rows)


def write_predictions(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['source_id', 'prediction'])
        writer.writerows((source, prediction) for source, _, prediction in rows)


def run_baseline(config, repo, resume=None, smoke=False):
    """Train/select only on validation; export every split sample and score via eval.py."""
    from dfd.teacher_data import VietnameseLmdb
    if resume:
        raise ValueError('Baseline resume is not supported; rerun from scratch')
    seed_everything(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_baseline(config).to(device)
    table_chars = ''.join(DecompositionTable.from_characters().characters[1:])
    if not set(table_chars).issubset(set(model.characters)):
        raise ValueError('Baseline vocabulary does not cover the shared project charset')
    root = Path(config['data']['root']).expanduser().resolve()
    train = VietnameseLmdb(root / 'lmdb/train', table_chars, training=True)
    val = VietnameseLmdb(root / 'lmdb/val', table_chars)
    train_loader = DataLoader(train, batch_size=config['training']['batch_size'], shuffle=True,
                              num_workers=config['training'].get('num_workers', 4), pin_memory=device.type == 'cuda')
    val_loader = DataLoader(val, batch_size=config['training']['batch_size'], num_workers=0)
    folder = prepare_run(config, repo)
    print(f'Run directory: {folder}', flush=True)
    def save_json(name, value):
        (folder / name).write_text(json.dumps(value, indent=2) + '\n')
    meta = json.loads((folder / 'metadata.json').read_text())
    summary = {'status': 'running', 'smoke_test': smoke, 'train_audit': train.audit, 'validation_audit': val.audit}
    meta['status'] = 'running'
    save_json('metadata.json', meta)
    save_json('training.json', summary)
    save_json('initialization.json', model.initialization)
    save_json('data_identity.json', {'freeze_sha256': digest(root / 'splits/freeze_manifest.json'),
                                    'lmdb_sha256': {s: digest(root / f'lmdb/{s}/data.mdb') for s in ('train', 'val')}})
    checkpoint_dir = folder / 'checkpoints'
    checkpoint_dir.mkdir()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'],
                                  weight_decay=config['training'].get('weight_decay', 0))
    epochs = 1 if smoke else config['training']['epochs']
    steps = 1 if smoke else len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['training']['lr'],
                    total_steps=epochs * steps, pct_start=config['training'].get('warmup_pct', .075)) if not smoke else None
    amp = device.type == 'cuda' and config['training'].get('amp', True)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    best_score, best_path = -1, None
    started = time.monotonic()
    try:
        for epoch in range(epochs):
            model.train()
            for index, (images, labels) in enumerate(train_loader):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, enabled=amp):
                    loss = model.loss(images.to(device), labels)
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite baseline loss')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 20.)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scheduler and scaler.get_scale() >= previous_scale:
                    scheduler.step()
                if smoke:
                    break
            rows = predict_loader(model, [next(iter(val_loader))] if smoke else val_loader, device)
            score = validation_accuracy(rows)
            print(f'Epoch {epoch + 1}: validation={score:.4%}, N={len(rows)}', flush=True)
            with (folder / 'validation.csv').open('a', newline='') as f:
                writer = csv.writer(f)
                if epoch == 0:
                    writer.writerow(['epoch', 'validation_word_acc'])
                writer.writerow([epoch + 1, score])
            if score > best_score:
                previous = best_path
                best_path = checkpoint_dir / f'best-epoch={epoch:02d}-validation_word_acc={score:.4f}.pt'
                torch.save({'state_dict': model.state_dict(), 'config': config, 'epoch': epoch,
                            'validation_word_acc': score, 'initialization': model.initialization}, best_path)
                best_score = score
                if previous and previous != best_path:
                    previous.unlink()
        if not smoke:
            model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)['state_dict'])
            prediction_dir = folder / 'baseline_predictions'
            for split in config['evaluation']['splits']:
                dataset = val if split == 'validation' else VietnameseLmdb(root / 'lmdb' / split, table_chars)
                loader = val_loader if split == 'validation' else DataLoader(dataset, batch_size=config['training']['batch_size'])
                write_predictions(prediction_dir / f'{split}.csv', predict_loader(model, loader, device))
            # Existing shared scorer keeps all validation/test samples and subset denominators.
            from eval import main as evaluate
            code = evaluate(['--run-dir', str(folder), '--predictions-dir', str(prediction_dir)])
            if code:
                raise RuntimeError(f'Baseline scoring failed with code {code}')
        summary.update(status='smoke_completed' if smoke else 'completed', best_checkpoint=str(best_path),
                       best_validation_word_acc=best_score)
    except BaseException as error:
        summary.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        raise
    finally:
        summary['elapsed_seconds'] = time.monotonic() - started
        meta['status'] = summary['status']
        save_json('training.json', summary)
        save_json('metadata.json', meta)
    return folder
