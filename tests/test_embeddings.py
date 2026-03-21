"""
Tests for the CLIP embedding layer.

These tests load the CLIP model (~350 MB, cached after first download) and are
therefore marked ``slow``.  Run the fast suite with::

    pytest -m "not slow"

Run only these tests with::

    pytest -m slow
"""

import uuid

import pytest
from PIL import Image

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def brown_image(tmp_path) -> str:
    img = Image.new("RGB", (320, 240), (180, 120, 80))
    path = tmp_path / "brown.jpg"
    img.save(str(path), "JPEG")
    return str(path)


@pytest.fixture()
def blue_image(tmp_path) -> str:
    img = Image.new("RGB", (320, 240), (20, 60, 200))
    path = tmp_path / "blue.jpg"
    img.save(str(path), "JPEG")
    return str(path)


@pytest.fixture()
def green_image(tmp_path) -> str:
    img = Image.new("RGB", (320, 240), (30, 180, 60))
    path = tmp_path / "green.jpg"
    img.save(str(path), "JPEG")
    return str(path)


# ---------------------------------------------------------------------------
# embed_image
# ---------------------------------------------------------------------------

class TestEmbedImage:
    def test_returns_list(self, brown_image):
        from litterbox.embeddings import embed_image
        assert isinstance(embed_image(brown_image), list)

    def test_correct_dimension(self, brown_image):
        from litterbox.embeddings import embed_image
        assert len(embed_image(brown_image)) == 512

    def test_values_are_floats(self, brown_image):
        from litterbox.embeddings import embed_image
        vec = embed_image(brown_image)
        assert all(isinstance(v, float) for v in vec)

    def test_not_all_zeros(self, brown_image):
        from litterbox.embeddings import embed_image
        assert any(v != 0.0 for v in embed_image(brown_image))

    def test_same_image_same_embedding(self, brown_image):
        from litterbox.embeddings import embed_image
        v1 = embed_image(brown_image)
        v2 = embed_image(brown_image)
        assert v1 == v2

    def test_different_images_different_embeddings(self, brown_image, blue_image):
        from litterbox.embeddings import embed_image
        assert embed_image(brown_image) != embed_image(blue_image)


# ---------------------------------------------------------------------------
# find_candidates — empty index
# ---------------------------------------------------------------------------

class TestFindCandidatesEmpty:
    def test_empty_index_returns_empty_list(self, brown_image):
        from litterbox.embeddings import find_candidates
        assert find_candidates(brown_image) == []


# ---------------------------------------------------------------------------
# add_to_index + find_candidates round-trip
# ---------------------------------------------------------------------------

class TestIndexRoundTrip:
    def test_added_image_is_found(self, brown_image):
        from litterbox.embeddings import add_to_index, find_candidates
        add_to_index(str(uuid.uuid4()), brown_image, "Whiskers", 1)
        results = find_candidates(brown_image)
        assert len(results) >= 1

    def test_correct_cat_name_returned(self, brown_image):
        from litterbox.embeddings import add_to_index, find_candidates
        add_to_index(str(uuid.uuid4()), brown_image, "Whiskers", 1)
        cat_name, _, _, _ = find_candidates(brown_image)[0]
        assert cat_name == "Whiskers"

    def test_correct_cat_id_returned(self, brown_image):
        from litterbox.embeddings import add_to_index, find_candidates
        add_to_index(str(uuid.uuid4()), brown_image, "Whiskers", 42)
        _, cat_id, _, _ = find_candidates(brown_image)[0]
        assert cat_id == 42

    def test_reference_path_returned(self, brown_image):
        from litterbox.embeddings import add_to_index, find_candidates
        add_to_index(str(uuid.uuid4()), brown_image, "Whiskers", 1)
        _, _, _, ref_path = find_candidates(brown_image)[0]
        assert brown_image in ref_path or ref_path == brown_image

    def test_result_is_list_of_tuples(self, brown_image):
        from litterbox.embeddings import add_to_index, find_candidates
        add_to_index(str(uuid.uuid4()), brown_image, "Whiskers", 1)
        results = find_candidates(brown_image)
        assert isinstance(results, list)
        assert len(results[0]) == 4  # (cat_name, cat_id, score, ref_path)


# ---------------------------------------------------------------------------
# Similarity scores
# ---------------------------------------------------------------------------

class TestSimilarityScores:
    def test_same_image_meets_threshold(self, brown_image):
        from litterbox.embeddings import add_to_index, find_candidates, ID_THRESHOLD
        add_to_index(str(uuid.uuid4()), brown_image, "Whiskers", 1)
        _, _, score, _ = find_candidates(brown_image)[0]
        assert score >= ID_THRESHOLD

    def test_same_image_near_one(self, brown_image):
        from litterbox.embeddings import add_to_index, find_candidates
        add_to_index(str(uuid.uuid4()), brown_image, "Whiskers", 1)
        _, _, score, _ = find_candidates(brown_image)[0]
        assert score > 0.99

    def test_different_image_below_threshold(self, brown_image, blue_image):
        from litterbox.embeddings import add_to_index, find_candidates, ID_THRESHOLD
        add_to_index(str(uuid.uuid4()), brown_image, "Whiskers", 1)
        _, _, score, _ = find_candidates(blue_image)[0]
        assert score < ID_THRESHOLD

    def test_scores_are_between_minus_one_and_one(self, brown_image, blue_image):
        from litterbox.embeddings import add_to_index, find_candidates
        add_to_index(str(uuid.uuid4()), brown_image, "Whiskers", 1)
        for result in find_candidates(blue_image):
            _, _, score, _ = result
            assert -1.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Ordering and n_results
# ---------------------------------------------------------------------------

class TestOrdering:
    def test_results_sorted_descending(self, brown_image, blue_image, green_image):
        from litterbox.embeddings import add_to_index, find_candidates
        add_to_index(str(uuid.uuid4()), brown_image, "Whiskers", 1)
        add_to_index(str(uuid.uuid4()), blue_image,  "Bluebell", 2)
        results = find_candidates(brown_image, n_results=2)
        assert len(results) == 2
        assert results[0][2] >= results[1][2]   # scores descending

    def test_n_results_limits_output(self, brown_image, blue_image, green_image):
        from litterbox.embeddings import add_to_index, find_candidates
        add_to_index(str(uuid.uuid4()), brown_image, "Whiskers", 1)
        add_to_index(str(uuid.uuid4()), blue_image,  "Bluebell", 2)
        add_to_index(str(uuid.uuid4()), green_image, "Greenie",  3)
        results = find_candidates(brown_image, n_results=2)
        assert len(results) == 2

    def test_n_results_one(self, brown_image, blue_image):
        from litterbox.embeddings import add_to_index, find_candidates
        add_to_index(str(uuid.uuid4()), brown_image, "Whiskers", 1)
        add_to_index(str(uuid.uuid4()), blue_image,  "Bluebell", 2)
        results = find_candidates(brown_image, n_results=1)
        assert len(results) == 1

    def test_top_result_is_same_cat(self, brown_image, blue_image):
        from litterbox.embeddings import add_to_index, find_candidates
        add_to_index(str(uuid.uuid4()), brown_image, "Whiskers", 1)
        add_to_index(str(uuid.uuid4()), blue_image,  "Bluebell", 2)
        cat_name, _, _, _ = find_candidates(brown_image, n_results=2)[0]
        assert cat_name == "Whiskers"
