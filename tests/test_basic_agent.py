"""
Tests for import-safe Bob helpers.
"""


def test_basic_agent_import_has_no_interactive_side_effects():
    """Importing the packaged module should not start the chat loop."""
    import basic_agent

    assert callable(basic_agent.main)
    assert basic_agent.web_search.name == "web_search"
    assert callable(basic_agent.web_search.invoke)


def test_litterbox_bob_entrypoint_uses_installed_module(monkeypatch):
    """The console entry point should not depend on a source-tree file path."""
    import basic_agent
    from litterbox import _cli

    called = []
    monkeypatch.setattr(basic_agent, "main", lambda: called.append(True))

    _cli.bob()

    assert called == [True]
