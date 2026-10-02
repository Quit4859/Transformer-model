from agentmodel.agent.loop import AgentLoop, LoopLimits
from agentmodel.agent.tools import WorkspaceTools


class FakeModel:
    def __init__(self):
        self.responses = iter(
            ['{"name":"read","arguments":{"path":"README.md"}}', "done"]
        )

    def generate(self, prompt, *, stop, max_new_tokens):
        return next(self.responses)


def test_agent_loop_dispatches_tool_then_returns_answer(tmp_path):
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")
    loop = AgentLoop(FakeModel(), WorkspaceTools(tmp_path))

    result = loop.run("inspect README", lambda context: str(context.messages))

    assert result == "done"
    assert len(loop.transcript) == 2
    assert loop.transcript[0].results[0]["result"] == "hello"


def test_agent_loop_uses_explicit_tool_allowlist(tmp_path):
    class Model:
        def generate(self, prompt, *, stop, max_new_tokens):
            return '{"name":"_path","arguments":{"relative":"README.md"}}'

    loop = AgentLoop(
        Model(),
        WorkspaceTools(tmp_path),
        LoopLimits(max_iterations=1),
    )

    assert loop.run("try private tool", lambda context: str(context.messages)).startswith("stopped:")
    assert loop.transcript[0].results[0]["error"] == "true"
