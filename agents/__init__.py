"""CrewAI integration package.

Disable optional telemetry before any CrewAI module can be imported.
"""

import os

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"
