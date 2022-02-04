import math

import numpy as np
import pytest

import mauve
from examples import load_gpt2_dataset
from mauve.compute_mauve import get_features_from_input


class TestMauve:
    @pytest.fixture(scope="class")
    def human_texts(self):
        return load_gpt2_dataset('data/amazon.valid.jsonl', num_examples=100)

    @pytest.fixture(scope="class")
    def generated_texts(self):
        return load_gpt2_dataset('data/amazon-xl-1542M.valid.jsonl', num_examples=100)

    @pytest.mark.parametrize(
        "batch_size",
        [16, 8, 4, 3, 2],
    )
    def test_batchify_mauve(self, human_texts, generated_texts, batch_size):
        out = mauve.compute_mauve(p_text=human_texts,
                                  q_text=generated_texts,
                                  device_id=0,
                                  max_text_length=256,
                                  batch_size=batch_size,
                                  verbose=False,
                                  use_float64=True)
        assert math.isclose(out.mauve, 0.99168, abs_tol=1e-4), f"{out.mauve} != 0.99168"

    def test_default_mauve(self, human_texts, generated_texts):
        out = mauve.compute_mauve(p_text=human_texts,
                                  q_text=generated_texts,
                                  device_id=0,
                                  max_text_length=256,
                                  verbose=False,
                                  use_float64=True)
        assert math.isclose(out.mauve, 0.99168, abs_tol=1e-4)

    @pytest.mark.parametrize(
        "batch_size",
        [16, 8, 4, 3, 2],
    )
    def test_batchify_mauve_feature_level(self, human_texts, batch_size):
        p_features_original = get_features_from_input(
            None, None, human_texts, 'gpt2-large', 1024,
            -1, name="p", verbose=False, batch_size=1, use_float64=True,
        )
        p_features_batched = get_features_from_input(
            None, None, human_texts, 'gpt2-large', 1024,
            -1, name="p", verbose=False, batch_size=batch_size, use_float64=True,
        )
        norm_of_difference = np.linalg.norm(p_features_original - p_features_batched, axis=1)  # shape = (n,)
        # ensure that new features are close to old features
        assert np.max(norm_of_difference) < 1e-5 * np.max(np.linalg.norm(p_features_original, axis=1))
