from pathlib import Path
from hashlib import sha256

import torch

import msa_lab.data.tiny_shakespeare as tiny_shakespeare
from msa_lab.data.tiny_shakespeare import iter_validation_batches, sample_next_token_batch


def test_load_tiny_shakespeare_text_caches_download(tmp_path: Path, monkeypatch) -> None:
    cache_path = tmp_path / "tiny_shakespeare" / "input.txt"
    downloaded_bytes = b"First Citizen:\\nHello.\\n"
    calls = 0

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            return None

        def read(self) -> bytes:
            return downloaded_bytes

    def fake_urlopen(url: str, *, timeout: int) -> FakeResponse:
        nonlocal calls
        assert url == tiny_shakespeare.TINY_SHAKESPEARE_URL
        assert timeout == 30
        calls += 1
        return FakeResponse()

    monkeypatch.setattr(tiny_shakespeare, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        tiny_shakespeare,
        "TINY_SHAKESPEARE_SHA256",
        sha256(downloaded_bytes).hexdigest(),
    )

    first_text = tiny_shakespeare.load_tiny_shakespeare_text(cache_path)
    second_text = tiny_shakespeare.load_tiny_shakespeare_text(cache_path)

    assert first_text == "First Citizen:\\nHello.\\n"
    assert second_text == first_text
    assert cache_path.read_bytes() == downloaded_bytes
    assert calls == 1


def test_sample_next_token_batch_is_shifted_and_reproducible() -> None:
    token_ids = torch.arange(10, dtype=torch.long)
    first_generator = torch.Generator(device="cpu").manual_seed(7)
    second_generator = torch.Generator(device="cpu").manual_seed(7)

    input_ids, labels = sample_next_token_batch(
        token_ids,
        batch_size=2,
        seq_len=4,
        generator=first_generator,
    )
    repeated_input_ids, repeated_labels = sample_next_token_batch(
        token_ids,
        batch_size=2,
        seq_len=4,
        generator=second_generator,
    )

    assert input_ids.shape == (2, 4)
    assert labels.shape == (2, 4)
    assert torch.equal(input_ids[:, 1:], input_ids[:, :-1] + 1)
    assert torch.equal(input_ids[:, 1:], labels[:, :-1])
    assert torch.equal(input_ids, repeated_input_ids)
    assert torch.equal(labels, repeated_labels)
    assert int(labels.max()) < len(token_ids)


def test_iter_validation_batches_is_ordered_and_non_overlapping() -> None:
    token_ids = torch.arange(14, dtype=torch.long)

    batches = list(iter_validation_batches(token_ids, batch_size=2, seq_len=4))

    assert len(batches) == 2
    first_inputs, first_labels = batches[0]
    final_inputs, final_labels = batches[1]
    assert first_inputs.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert first_labels.tolist() == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert final_inputs.tolist() == [[8, 9, 10, 11]]
    assert final_labels.tolist() == [[9, 10, 11, 12]]
    assert torch.equal(torch.cat((first_labels.flatten(), final_labels.flatten())), torch.arange(1, 13))
