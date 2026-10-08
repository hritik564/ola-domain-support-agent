"""CrewAI tool base import with a test-only fallback when CrewAI is blocked."""

try:
    from crewai.tools import BaseTool as CrewAIBaseTool

    CREWAI_TOOL_AVAILABLE = True
except ImportError:
    CREWAI_TOOL_AVAILABLE = False

    class CrewAIBaseTool:
        """Minimal stand-in for direct unit tests; never used to run a Crew."""

        name: str
        description: str
        args_schema: type

        def run(self, **kwargs):
            return self._run(**kwargs)

        def _run(self, **kwargs):
            raise NotImplementedError
