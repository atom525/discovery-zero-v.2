from dz_hypergraph.tools.external_prm import ExternalPRM, ExternalPRMConfig


def test_external_prm_extracts_numeric_score_from_text():
    prm = ExternalPRM(ExternalPRMConfig(provider="chat_completion", model="dummy"))
    score = prm._extract_score({"choices": [{"text": "0.73"}]})
    assert score == 0.73


def test_external_prm_enabled_for_chat_completion_provider():
    prm = ExternalPRM(ExternalPRMConfig(provider="chat_completion", model="dummy"))
    assert prm.enabled is True
