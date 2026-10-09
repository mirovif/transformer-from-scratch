import copy
from pathlib import Path
import hashlib
import json
import pytest
import torch
from transformer import Transformer, MultiHeadAttention
from transformer.embedding import InputEmbedding
from transformer.masks import causal_mask, padding_mask
from transformer.copy_task import batch_tensors, make_examples, PAD

torch.set_num_threads(2)


def small_model(dropout=0.):
    torch.manual_seed(7)
    return Transformer(src_vocab=24, tgt_vocab=24, D=16, heads=4,
                       d_ff=32, N=2, max_len=16, dropout=dropout)


def test_different_source_target_lengths_and_tensor_shape():
    model = small_model().eval()
    output = model(torch.tensor([[3, 4, 5, 2, 0], [6, 7, 2, 0, 0]]),
                   torch.tensor([[1, 3, 4], [1, 6, 7]]))
    assert output.shape == (2, 3, 24)
    assert torch.isfinite(output).all()


def test_automatic_causal_mask_blocks_future_token_changes():
    model = small_model().eval()
    source = torch.tensor([[3, 4, 5, 2]])
    target = torch.tensor([[1, 3, 4, 5]])
    changed = target.clone(); changed[0, -1] = 12
    torch.testing.assert_close(model(source, target)[:, :3], model(source, changed)[:, :3])


def test_attention_matches_independent_scaled_dot_product_reference():
    torch.manual_seed(4)
    attention = MultiHeadAttention(D=8, heads=2)
    query, key, value = torch.randn(2, 3, 8), torch.randn(2, 5, 8), torch.randn(2, 5, 8)
    def split(projected):
        return projected.reshape(2, -1, 2, 4).transpose(1, 2)
    q, k, v = split(attention.Q(query)), split(attention.K(key)), split(attention.V(value))
    scores = torch.einsum('bhqd,bhkd->bhqk', q, k) / 2.
    expected = torch.einsum('bhqk,bhkd->bhqd', scores.softmax(-1), v)
    expected = attention.W_o(expected.transpose(1, 2).reshape(2, 3, 8))
    torch.testing.assert_close(attention(query, key, value), expected)


@pytest.mark.parametrize('rank', [2, 3, 4])
def test_masks_are_reusable_without_mutating_shape_or_values(rank):
    attention = MultiHeadAttention(D=8, heads=2)
    tensor = torch.randn(2, 3, 8)
    mask = causal_mask(3)
    if rank == 2:
        mask = mask[0, 0]
    elif rank == 3:
        mask = mask[:, 0].expand(2, -1, -1)
    before = mask.clone(); shape = mask.shape
    first = attention(tensor, tensor, tensor, mask)
    second = attention(tensor, tensor, tensor, mask)
    assert mask.shape == shape
    torch.testing.assert_close(mask, before)
    torch.testing.assert_close(first, second)


def test_padding_keys_have_no_effect_on_output():
    attention = MultiHeadAttention(D=8, heads=2)
    query, keys, values = torch.randn(1, 3, 8), torch.randn(1, 4, 8), torch.randn(1, 4, 8)
    mask = torch.tensor([[[[True, True, False, False]]]])
    changed_keys, changed_values = keys.clone(), values.clone()
    changed_keys[:, 2:] = 1000; changed_values[:, 2:] = -1000
    torch.testing.assert_close(attention(query, keys, values, mask),
                               attention(query, changed_keys, changed_values, mask))


def test_padding_length_does_not_change_real_token_logits():
    model = small_model().eval()
    target = torch.tensor([[1, 3, 4]])
    compact = model(torch.tensor([[3, 4, 2]]), target)
    padded = model(torch.tensor([[3, 4, 2, 0, 0]]), target)
    torch.testing.assert_close(compact, padded, rtol=1e-5, atol=1e-6)


def test_completely_masked_rows_have_finite_forward_and_backward():
    attention = MultiHeadAttention(D=8, heads=2)
    tensor = torch.randn(1, 3, 8, requires_grad=True)
    output = attention(tensor, tensor, tensor, torch.zeros(3, 3, dtype=torch.bool))
    output.sum().backward()
    assert torch.isfinite(output).all() and torch.isfinite(tensor.grad).all()


def test_model_has_finite_gradients_through_encoder_decoder_and_ffn():
    model = small_model()
    source, target, labels = batch_tensors([(3, 4, 5), (6, 7, 8, 9)])
    loss = torch.nn.functional.cross_entropy(model(source, target).reshape(-1, 24), labels.reshape(-1), ignore_index=PAD)
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_state_dict_round_trip_and_eval_determinism():
    model = small_model(.2).eval()
    clone = copy.deepcopy(model)
    clone.load_state_dict(model.state_dict())
    source, target, _ = batch_tensors([(3, 4, 5)])
    torch.testing.assert_close(model(source, target), clone(source, target))
    torch.testing.assert_close(model(source, target), model(source, target))
    assert 'src_input.pe' in model.state_dict()


def test_embedding_dropout_and_buffer_device():
    embedding = InputEmbedding(24, 9, 12, dropout=.5)
    tokens = torch.tensor([[3, 4, 5]])
    embedding.eval()
    torch.testing.assert_close(embedding(tokens), embedding(tokens))
    embedding.train()
    assert not torch.equal(embedding(tokens), embedding(tokens))
    assert embedding.pe.device == embedding.embedding.weight.device
    assert embedding(tokens).shape == (1, 3, 9)  # odd D in positional encoding


@pytest.mark.parametrize('D,heads', [(8, 3), (8, 0), (0, 2)])
def test_invalid_head_count_is_rejected(D, heads):
    with pytest.raises(ValueError):
        MultiHeadAttention(D, heads)


def test_invalid_masks_and_sequence_length_are_rejected():
    attention = MultiHeadAttention(8, 2)
    tensor = torch.randn(1, 3, 8)
    with pytest.raises(TypeError):
        attention(tensor, tensor, tensor, torch.ones(3, 3))
    with pytest.raises(ValueError):
        attention(tensor, tensor, tensor, torch.ones(7, 7, dtype=torch.bool))
    with pytest.raises(ValueError):
        InputEmbedding(24, 8, 2, .1)(torch.tensor([[1, 3, 4]]))


def test_dataset_splits_are_disjoint_and_shift_is_correct():
    train = make_examples(30, 42)
    validation = make_examples(10, 43, excluded=train)
    assert set(train).isdisjoint(validation)
    _, target, labels = batch_tensors([(3, 4, 5)])
    assert target.tolist() == [[1, 3, 4, 5]]
    assert labels.tolist() == [[3, 4, 5, 2]]


def test_tiny_batch_can_be_overfit():
    torch.manual_seed(42)
    model = Transformer(24, 24, 16, 4, 32, 1, 16, 0.)
    source, target, labels = batch_tensors([(3, 4, 5), (6, 7, 8)])
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    initial = None
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(source, target).reshape(-1, 24), labels.reshape(-1))
        if initial is None:
            initial = loss.item()
        loss.backward(); optimizer.step()
    assert loss.item() < initial * .1
    assert (model(source, target).argmax(-1) == labels).all()


def test_original_files_are_preserved_byte_for_byte():
    folder = Path(__file__).resolve().parents[1] / 'docs/original'
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert len(manifest['files']) == 9
    for name, expected in manifest['files'].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected
