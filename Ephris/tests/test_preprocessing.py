"""Feature typing and checkpoint-compatible deterministic permutation."""

import pytest
import torch

from ephris.data.permutation import make_inference_permutation
from ephris.data.preprocessing import FeaturePreprocessor


def test_mixed_feature_types_and_transforms():
    x = torch.tensor(
        [
            [0.1, 2.0, 1.0, 7.0],
            [0.3, 8.0, 2.0, 7.0],
            [0.5, 2.0, 3.0, 7.0],
            [0.7, 8.0, 2.0, 7.0],
        ]
    )
    processor = FeaturePreprocessor()
    processor.prepare(x)
    assert processor.feature_metadata().tolist() == [0, 1, 2, 0]
    actual = processor.transform(x)
    expected = x.clone()
    expected[:, 0] = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std(unbiased=False)
    expected[:, 1] = torch.tensor([0.0, 1.0, 0.0, 1.0])
    expected[:, 2] = torch.tensor([-1.0, 0.0, 1.0, 0.0])
    expected[:, 3] = 0
    torch.testing.assert_close(actual, expected)
    assert torch.equal(processor.transform(x), actual)


@pytest.mark.parametrize(
    "features,classes,order,labels",
    [
        (1, 2, None, [1, 0]),
        (13, 5, [2, 12, 3, 6, 11, 5, 4, 7, 1, 0, 8, 9, 10], [2, 0, 1, 4, 3]),
        (
            32,
            10,
            [
                16,
                0,
                27,
                8,
                7,
                23,
                5,
                4,
                14,
                31,
                10,
                2,
                9,
                22,
                19,
                13,
                11,
                29,
                12,
                20,
                18,
                6,
                3,
                1,
                30,
                24,
                17,
                25,
                21,
                26,
                28,
                15,
            ],
            [0, 9, 2, 4, 7, 8, 6, 3, 1, 5],
        ),
    ],
)
def test_permutations_match_reference(features, classes, order, labels):
    result = make_inference_permutation(x=torch.zeros(4, features), num_outputs=classes)
    assert (
        None if result.feature_order is None else result.feature_order.tolist()
    ) == order
    assert result.class_permutation.tolist() == labels
