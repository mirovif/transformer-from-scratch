"""Train the imported model on a fixed copy task, with separate validation and test."""
import argparse
import csv
import json
import platform
from pathlib import Path
import random
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn

from .copy_task import PAD, VOCAB_SIZE, make_examples, batch_tensors, greedy_decode, generated_sequence
from .model import Transformer

ROOT = Path(__file__).resolve().parents[2]
CONFIG = dict(src_vocab=VOCAB_SIZE, tgt_vocab=VOCAB_SIZE, D=32, heads=4,
              d_ff=64, N=2, max_len=16, dropout=.1, pad_id=PAD)


@torch.no_grad()
def evaluate(model, examples, device):
    model.eval()
    source, decoder_input, labels = batch_tensors(examples, device)
    logits = model(source, decoder_input)
    total_loss = nn.functional.cross_entropy(logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1),
                                            ignore_index=PAD, reduction='sum')
    valid = labels != PAD
    token_accuracy = ((logits.argmax(-1) == labels) & valid).sum() / valid.sum()
    generated = greedy_decode(model, source)
    matches, samples = [], []
    for sequence, row in zip(examples, generated):
        predicted, emitted_eos = generated_sequence(row)
        correct = predicted == list(sequence) and emitted_eos
        matches.append(correct)
        samples.append({'input': list(sequence), 'expected': list(sequence),
                        'generated': predicted, 'emitted_eos': emitted_eos, 'exact_match': correct})
    return {'loss': float(total_loss / valid.sum()),
            'teacher_forced_token_accuracy': float(token_accuracy),
            'greedy_exact_match': sum(matches) / len(matches)}, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=2400)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test-seed', type=int, default=45)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error('--steps must be positive')
    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    rng = random.Random(args.seed)
    train = make_examples(1024, args.seed)
    validation = make_examples(128, args.seed + 1, excluded=train)
    development_holdout = make_examples(128, args.seed + 2, excluded=[*train, *validation])
    test = make_examples(128, args.test_seed, excluded=[*train, *validation, *development_holdout])
    assert set(train).isdisjoint(validation) and set(train).isdisjoint(test)
    assert set(validation).isdisjoint(test)
    for folder in ['reports', 'images', 'checkpoints', 'data']:
        (ROOT / folder).mkdir(exist_ok=True)
    (ROOT / 'data/copy_splits.json').write_text(json.dumps(
        {'train': train, 'validation': validation, 'test': test}), encoding='utf-8')
    model = Transformer(**CONFIG).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    initial, _ = evaluate(model, validation, args.device)
    history = [{'step': 0, 'train_loss': None, 'validation_loss': initial['loss'],
                'validation_token_accuracy': initial['teacher_forced_token_accuracy'],
                'validation_greedy_exact_match': initial['greedy_exact_match']}]
    best_loss, best_step = float('inf'), 0
    print('Initial validation loss:', round(initial['loss'], 4), flush=True)
    for step in range(1, args.steps + 1):
        model.train()
        sampled = rng.sample(train, 64)
        source, decoder_input, labels = batch_tensors(sampled, args.device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(source, decoder_input)
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
        optimizer.step()
        if step % 50 == 0 or step == args.steps:
            metrics, _ = evaluate(model, validation, args.device)
            history.append({'step': step, 'train_loss': float(loss.detach()),
                            'validation_loss': metrics['loss'],
                            'validation_token_accuracy': metrics['teacher_forced_token_accuracy'],
                            'validation_greedy_exact_match': metrics['greedy_exact_match']})
            if metrics['loss'] < best_loss:
                best_loss, best_step = metrics['loss'], step
                torch.save({'config': CONFIG, 'state_dict': model.state_dict()}, ROOT / 'checkpoints/copy_model.pt')
            print(f"Step {step:4} train_loss={loss.item():.4f} val_loss={metrics['loss']:.4f} "
                  f"val_exact={metrics['greedy_exact_match']:.3f}", flush=True)
    checkpoint = torch.load(ROOT / 'checkpoints/copy_model.pt', map_location=args.device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    validation_metrics, _ = evaluate(model, validation, args.device)
    # The test set is evaluated only after checkpoint selection on validation loss.
    test_metrics, test_samples = evaluate(model, test, args.device)
    with (ROOT / 'reports/training_history.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader(); writer.writerows(history)
    summary = {'task': 'Synthetic sequence copy; not a real-language benchmark', 'seed': args.seed,
               'test_seed': args.test_seed,
               'development_protocol': 'Duration extended after a 600-step run because validation loss was still falling. '
                                       'The earlier holdout (seed 44) is excluded from this fresh final test (seed 45).',
               'device': args.device, 'steps': args.steps, 'selected_step': best_step,
               'checkpoint_selection': 'minimum validation token cross-entropy',
               'train_sequences': len(train), 'validation_sequences': len(validation), 'test_sequences': len(test),
               'config': CONFIG, 'parameters': sum(parameter.numel() for parameter in model.parameters()),
               'initial_validation': initial, 'validation': validation_metrics, 'test': test_metrics,
               'optimizer': 'Adam, lr=0.001; batch_size=64; clip_grad_norm=1.0',
               'torch': torch.__version__, 'python': platform.python_version(),
               'elapsed_seconds': round(time.perf_counter() - start, 2)}
    (ROOT / 'reports/summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    (ROOT / 'reports/test_generations.json').write_text(json.dumps(test_samples, indent=2) + '\n', encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    steps = [row['step'] for row in history]
    axes[0].plot(steps[1:], [row['train_loss'] for row in history[1:]], label='Training batch', color='#a4bbc3')
    axes[0].plot(steps, [row['validation_loss'] for row in history], label='Validation', color='#256b85')
    axes[0].set(xlabel='Optimizer step', ylabel='Token cross-entropy', title='Learning curve · copy task')
    axes[0].legend()
    axes[1].plot(steps, [row['validation_token_accuracy'] for row in history], label='Teacher-forced token', color='#a4bbc3')
    axes[1].plot(steps, [row['validation_greedy_exact_match'] for row in history], label='Greedy sequence exact match', color='#256b85')
    axes[1].set(xlabel='Optimizer step', ylabel='Accuracy', ylim=(0, 1.03), title='Validation · two evaluation modes')
    axes[1].legend(loc='lower right', fontsize=8)
    for axis in axes:
        axis.spines[['right', 'top']].set_visible(False)
        axis.grid(alpha=.18)
    fig.tight_layout()
    fig.savefig(ROOT / 'images/training_curve.png', dpi=170, bbox_inches='tight')
    plt.close(fig)
    print('Final test:', json.dumps(test_metrics), flush=True)


if __name__ == '__main__':
    main()
