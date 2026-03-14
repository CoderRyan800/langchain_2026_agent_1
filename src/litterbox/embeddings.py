from pathlib import Path
from typing import List, Tuple, Optional

import chromadb
from PIL import Image
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).parent.parent.parent
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma"
ID_THRESHOLD = 0.82

# Module-level singletons — loaded once on first use
_model: Optional[SentenceTransformer] = None
_collection: Optional[chromadb.Collection] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        # ~350 MB download on first run; cached afterwards
        _model = SentenceTransformer("clip-ViT-B-32")
    return _model


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _collection = client.get_or_create_collection(
            name="cat_images",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def embed_image(image_path: str) -> List[float]:
    """Return a CLIP embedding for the image at image_path."""
    img = Image.open(image_path).convert("RGB")
    return _get_model().encode(img).tolist()


def add_to_index(chroma_id: str, image_path: str, cat_name: str, cat_id: int) -> None:
    """Embed an image and add it to the Chroma collection."""
    embedding = embed_image(image_path)
    _get_collection().add(
        ids=[chroma_id],
        embeddings=[embedding],
        metadatas=[{"cat_name": cat_name, "cat_id": cat_id, "file_path": image_path}],
    )


def find_candidates(
    image_path: str, n_results: int = 3
) -> List[Tuple[str, int, float, str]]:
    """
    Nearest-neighbor search against stored cat reference images.

    Returns a list of (cat_name, cat_id, similarity_score, reference_image_path)
    sorted by descending similarity.  Chroma returns cosine *distance*
    (0 = identical, 2 = opposite), so similarity = 1 - distance.
    """
    collection = _get_collection()
    if collection.count() == 0:
        return []

    embedding = embed_image(image_path)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(n_results, collection.count()),
    )

    candidates = []
    for i, _id in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        similarity = 1.0 - distance
        candidates.append((
            meta["cat_name"],
            int(meta["cat_id"]),
            similarity,
            meta["file_path"],
        ))

    return candidates
