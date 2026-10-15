"""Stage B tests for task outcome recording and working-memory cleanup."""

from codepilot.agents.execution_engine import ExecutionStage
from codepilot.agents.orchestrator import CodePilotOrchestrator
from codepilot.github.models import GitHubIssue
from codepilot.graph.lifecycle import TaskStatus


class DummyAgent:
    def invoke(self, payload):
        return payload


class FakeEpisodicMemory:
    def __init__(self):
        self.records = []

    def get_recent_tasks(self, *, limit, repository=None):
        return []

    def record_task(self, **record):
        self.records.append(record)
        return record


class FakeSemanticMemory:
    def retrieve_lessons(self, **kwargs):
        return []


def make_issue():
    return GitHubIssue(
        number=42,
        title="Fix login crash",
        body="The application crashes during login.",
        url="https://github.com/example/repo/issues/42",
    )


def make_orchestrator(stages):
    episodic = FakeEpisodicMemory()
    orchestrator = CodePilotOrchestrator(
        model=object(),
        repository="example/repo",
        agent_factory=lambda **kwargs: DummyAgent(),
        episodic_memory=episodic,
        semantic_memory=FakeSemanticMemory(),
        execution_stages_factory=lambda task: stages(task),
    )
    return orchestrator, episodic


def test_done_task_is_recorded_and_working_memory_is_cleared():
    captured = {}

    def stages(task):
        captured["working_memory"] = task["working_memory"]
        return [
            ExecutionStage("explore", TaskStatus.EXPLORING, lambda: "repository inspected"),
            ExecutionStage("implement", TaskStatus.IMPLEMENTING, lambda: "fix applied"),
            ExecutionStage("test", TaskStatus.TESTING, lambda: "tests passed"),
            ExecutionStage("open_pr", TaskStatus.PR_OPENED, lambda: "PR opened"),
            ExecutionStage("finish", TaskStatus.DONE, lambda: "completed"),
        ]

    orchestrator, episodic = make_orchestrator(stages)
    result = orchestrator.execute(make_issue())

    memory = captured["working_memory"]
    assert result.lifecycle_status == TaskStatus.DONE
    assert memory.status == "DONE"
    assert memory.issue == {}
    assert memory.repository_map == {}
    assert memory.episodic_context == []
    assert memory.semantic_context == []
    assert len(episodic.records) == 1
    assert episodic.records[0]["status"] == "DONE"
    assert episodic.records[0]["issue_number"] == 42
    assert "tests passed" in episodic.records[0]["summary"]


def test_failed_task_is_recorded_and_working_memory_is_cleared():
    captured = {}

    def stages(task):
        captured["working_memory"] = task["working_memory"]

        def fail():
            raise RuntimeError("test command failed")

        return [
            ExecutionStage("explore", TaskStatus.EXPLORING, lambda: "mapped"),
            ExecutionStage("implement", TaskStatus.IMPLEMENTING, fail),
        ]

    orchestrator, episodic = make_orchestrator(stages)
    result = orchestrator.execute(make_issue())

    memory = captured["working_memory"]
    assert result.lifecycle_status == TaskStatus.FAILED
    assert memory.status == "FAILED"
    assert memory.issue == {}
    assert memory.repository_map == {}
    assert len(episodic.records) == 1
    assert episodic.records[0]["status"] == "FAILED"
    assert "test command failed" in episodic.records[0]["summary"]


def test_nonterminal_execution_keeps_working_memory_and_does_not_record_outcome():
    captured = {}

    def stages(task):
        captured["working_memory"] = task["working_memory"]
        return [ExecutionStage("explore", TaskStatus.EXPLORING, lambda: "mapped")]

    orchestrator, episodic = make_orchestrator(stages)
    result = orchestrator.execute(make_issue())

    memory = captured["working_memory"]
    assert result.lifecycle_status == TaskStatus.EXPLORING
    assert memory.status == "EXPLORING"
    assert memory.issue["title"] == "Fix login crash"
    assert len(episodic.records) == 0
