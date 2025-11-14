from __future__ import annotations

from ati_engine.taxonomy.loader import TaxonomyLoader


def test_taxonomy_labels_list():
    loader = TaxonomyLoader("ati_engine/taxonomy/sample_taxonomy.yaml")
    data = loader.load()
    labels = loader.list_labels(data)
    assert any("Food" in l for l in labels)
    assert isinstance(labels, list)
    assert len(labels) > 0
