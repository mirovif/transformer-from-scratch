"""A deliberately synthetic copy task for checking the author's architecture."""
import random
import torch

PAD, BOS, EOS = 0, 1, 2
VOCAB_SIZE = 24


def make_examples(count, seed, excluded=()):
    rng = random.Random(seed)
    seen = set(excluded)
    examples = []
    while len(examples) < count:
        sequence = tuple(rng.randrange(3, VOCAB_SIZE) for _ in range(rng.randint(3, 7)))
        if sequence not in seen:
            seen.add(sequence)
            examples.append(sequence)
    return examples


def batch_tensors(examples, device='cpu'):
    max_length = max(len(sequence) for sequence in examples) + 1
    source = torch.full((len(examples), max_length), PAD, dtype=torch.long, device=device)
    decoder_input = source.clone()
    labels = source.clone()
    for row, sequence in enumerate(examples):
        length = len(sequence)
        source[row, :length] = torch.tensor(sequence, device=device)
        source[row, length] = EOS
        decoder_input[row, 0] = BOS
        decoder_input[row, 1:length + 1] = torch.tensor(sequence, device=device)
        labels[row, :length] = torch.tensor(sequence, device=device)
        labels[row, length] = EOS
    return source, decoder_input, labels


@torch.no_grad()
def greedy_decode(model, source, max_new_tokens=9):
    model.eval()
    output = torch.full((source.shape[0], 1), BOS, dtype=torch.long, device=source.device)
    finished = torch.zeros(source.shape[0], dtype=torch.bool, device=source.device)
    for _ in range(max_new_tokens):
        predicted = model(source, output)[:, -1].argmax(dim=-1)
        predicted = torch.where(finished, PAD, predicted)
        output = torch.cat((output, predicted.unsqueeze(1)), dim=1)
        finished |= predicted == EOS
        if finished.all():
            break
    return output


def generated_sequence(row):
    tokens = row.tolist()[1:]  # drop BOS
    if EOS in tokens:
        return tokens[:tokens.index(EOS)], True
    return [token for token in tokens if token != PAD], False
