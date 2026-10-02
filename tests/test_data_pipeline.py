from agentmodel.data.mixture import Mixture, MixtureSampler
from agentmodel.data.packing import pack_documents
from agentmodel.data.quality import benchmark_decontaminate


def test_benchmark_decontamination_removes_high_overlap_documents():
    benchmark = ["def solve(value): return value + 1"]
    documents = [
        "def solve(value):   return value + 1",
        "def unrelated(value): return value * 2",
    ]

    result = benchmark_decontaminate(documents, benchmark, ngram_size=8, threshold=0.8)

    assert result.contaminated_indices == [0]
    assert result.documents == [documents[1]]
    assert result.overlap_scores[0] >= 0.8


def test_mixture_sampling_is_reproducible_and_weighted():
    mixtures = [Mixture("code", 9.0), Mixture("text", 1.0)]
    first = MixtureSampler(mixtures, seed=42).sample(1000)
    second = MixtureSampler(mixtures, seed=42).sample(1000)

    assert first == second
    assert first.count("code") > first.count("text")


def test_packing_masks_next_token_loss_at_document_boundaries():
    batch = pack_documents([[10, 11], [20, 21]], seq_len=4, pad_token_id=0)

    assert batch.input_ids.tolist() == [[10, 11, 20, 21]]
    assert batch.targets.tolist() == [[11, -100, 21, -100]]
