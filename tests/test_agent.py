import pytest

from agentmodel.agent.context import Context
from agentmodel.agent.loop import AgentLoop, LoopLimits, truncate
from agentmodel.agent.parser import ToolCallParser
from agentmodel.agent.tools import PermissionError, WorkspaceTools


class ScriptedModel:
    """Replays a fixed list of assistant turns."""

    def __init__(self, turns: list[str]) -> None:
        self.turns = list(turns)
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, stop: tuple[str, ...], max_new_tokens: int) -> str:
        self.prompts.append(prompt)
        return self.turns.pop(0) if self.turns else "done"


def render(context: Context) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in context.messages)


def test_streaming_tool_parser_handles_split_json():
    parser = ToolCallParser()
    assert parser.feed('{"name":"read","arguments":') == []
    calls = parser.feed('{"path":"README.md"}}')
    assert calls[0].name == "read"
    assert calls[0].arguments == {"path": "README.md"}
    parser.finish()


def test_parser_emits_multiple_calls_from_one_chunk():
    parser = ToolCallParser()
    calls = parser.feed('{"name":"read","arguments":{"path":"a"}}{"name":"read"}')
    assert [c.name for c in calls] == ["read", "read"]
    parser.finish()


def test_parser_rejects_truncated_call():
    parser = ToolCallParser()
    parser.feed('{"name":"read","arguments":')
    with pytest.raises(ValueError):
        parser.finish()


def test_parser_rejects_missing_name():
    parser = ToolCallParser()
    with pytest.raises(ValueError):
        parser.feed('{"arguments":{}}')
        parser.finish()


def test_workspace_tools_enforce_permissions_and_root(tmp_path):
    tools = WorkspaceTools(tmp_path)
    with pytest.raises(PermissionError):
        tools.write("x.txt", "x")
    with pytest.raises(ValueError):
        tools.read("../outside")


def test_workspace_tools_edit_and_diff(tmp_path):
    tools = WorkspaceTools(tmp_path, allow_write=True)
    tools.write("a.txt", "one\ntwo\n")
    tools.edit("a.txt", "one", "ONE")
    assert tools.read("a.txt") == "ONE\ntwo\n"
    diff = tools.diff("a.txt", "one\ntwo\n", "ONE\ntwo\n")
    assert "-one" in diff and "+ONE" in diff


def test_workspace_tools_grep_and_glob(tmp_path):
    tools = WorkspaceTools(tmp_path, allow_write=True)
    tools.write("pkg/mod.py", "def needle():\n    pass\n")
    assert tools.glob("**/*.py") == ["pkg/mod.py"]
    assert any("needle" in hit for hit in tools.grep("needle"))


def test_bash_requires_permission(tmp_path):
    with pytest.raises(PermissionError):
        WorkspaceTools(tmp_path).bash("echo hi")
    out = WorkspaceTools(tmp_path, allow_bash=True).bash("echo hi")
    assert out["returncode"] == 0 and "hi" in out["stdout"]


def test_context_compacts_old_messages():
    context = Context([], max_messages=4)
    for index in range(8):
        context.append("user", str(index))
    assert len(context.messages) <= 4
    assert context.messages[0]["role"] == "system"


def test_truncate_reports_dropped_characters():
    assert truncate("abc", 10) == "abc"
    out = truncate("x" * 20, 5)
    assert out.startswith("xxxxx")
    assert "15 characters truncated" in out


def test_agent_loop_runs_tool_then_returns_final_text(tmp_path):
    tools = WorkspaceTools(tmp_path, allow_write=True)
    tools.write("a.txt", "hello")
    model = ScriptedModel(['{"name":"read","arguments":{"path":"a.txt"}}', "the file says hello"])
    loop = AgentLoop(model, tools, LoopLimits(max_iterations=4))

    result = loop.run("read a.txt", render)

    assert result == "the file says hello"
    assert len(loop.transcript) == 2
    assert loop.transcript[0].calls[0].name == "read"
    assert "hello" in loop.transcript[0].results[0]["result"]


def test_agent_loop_surfaces_tool_errors_back_into_context(tmp_path):
    tools = WorkspaceTools(tmp_path)
    model = ScriptedModel(
        ['{"name":"write","arguments":{"path":"a.txt","content":"x"}}', "cannot write"]
    )
    loop = AgentLoop(model, tools, LoopLimits(max_iterations=4))

    result = loop.run("write a.txt", render)

    assert result == "cannot write"
    assert loop.transcript[0].results[0]["error"] == "true"
    assert "PermissionError" in loop.transcript[0].results[0]["result"]


def test_agent_loop_reports_unknown_tool(tmp_path):
    model = ScriptedModel(['{"name":"teleport"}', "no such tool"])
    loop = AgentLoop(model, WorkspaceTools(tmp_path), LoopLimits(max_iterations=4))
    loop.run("teleport", render)
    assert "unknown tool" in loop.transcript[0].results[0]["result"]


def test_agent_loop_feeds_malformed_call_back(tmp_path):
    model = ScriptedModel(['{"name":"read","arguments":', "recovered"])
    loop = AgentLoop(model, WorkspaceTools(tmp_path), LoopLimits(max_iterations=4))
    assert loop.run("do a thing", render) == "recovered"
    # the malformed turn did not produce a call, so the next turn is treated as final
    assert loop.transcript[0].calls == []
    assert model.prompts[-1] != model.prompts[0]


def test_agent_loop_stops_at_iteration_budget(tmp_path):
    model = ScriptedModel(['{"name":"read","arguments":{"path":"a.txt"}}'] * 10)
    loop = AgentLoop(model, WorkspaceTools(tmp_path), LoopLimits(max_iterations=3))
    result = loop.run("loop forever", render)
    assert result == "stopped: iteration budget exhausted"
    assert len(loop.transcript) == 3


def test_agent_loop_truncates_large_tool_output(tmp_path):
    tools = WorkspaceTools(tmp_path, allow_write=True)
    tools.write("big.txt", "y" * 5000)
    model = ScriptedModel(['{"name":"read","arguments":{"path":"big.txt"}}', "ok"])
    loop = AgentLoop(model, tools, LoopLimits(max_iterations=4, max_tool_output_chars=100))
    loop.run("read big", render)
    assert "characters truncated" in loop.transcript[0].results[0]["result"]