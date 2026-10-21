import pytest

from src.aggregation import (
    MajorityVotingAggregator,
    MaxScoreAggregator,
    MeanScoreAggregator,
    MedianScoreAggregator,
    MinScoreAggregator,
    PercentileScoreAggregator,
)


SCORES = [0.1, 0.4, 0.8, 0.9]


def test_score_aggregators_return_expected_values() -> None:
    # Cada estratégia deve retornar o valor matemático esperado para os mesmos scores.
    assert MeanScoreAggregator().aggregate(SCORES) == pytest.approx(0.55)
    assert MedianScoreAggregator().aggregate(SCORES) == pytest.approx(0.6)
    assert PercentileScoreAggregator(75).aggregate(SCORES) == pytest.approx(0.825)
    assert MaxScoreAggregator().aggregate(SCORES) == pytest.approx(0.9)
    assert MinScoreAggregator().aggregate(SCORES) == pytest.approx(0.1)


def test_majority_voting_returns_fraction_of_positive_votes() -> None:
    # Dois dos quatro scores estão acima do limiar de 0,5, portanto o resultado é 0,5.
    score = MajorityVotingAggregator(0.5).aggregate(SCORES)

    assert score == pytest.approx(0.5)


@pytest.mark.parametrize(
    "aggregator",
    [
        MeanScoreAggregator(),
        MedianScoreAggregator(),
        PercentileScoreAggregator(75),
        MaxScoreAggregator(),
        MinScoreAggregator(),
        MajorityVotingAggregator(0.5),
    ],
)
def test_aggregators_reject_empty_scores(aggregator) -> None:
    # Não é possível gerar um score agregado sem ao menos um detector.
    with pytest.raises(ValueError):
        aggregator.aggregate([])


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_majority_voting_validates_threshold(threshold: float) -> None:
    # O limiar representa um score e precisa permanecer no intervalo de 0 a 1.
    with pytest.raises(ValueError):
        MajorityVotingAggregator(threshold)
