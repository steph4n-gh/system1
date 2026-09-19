import re

with open("README.md", "r") as f:
    text = f.read()

# Replace main references
replacements = {
    "Why Reflex": "Why System 1",
    "Without Reflex": "Without System 1",
    "With Reflex": "With System 1",
    "Reflex retains": "System 1 retains",
    "Reflex is the on-metal": "System 1 is the on-metal",
    "Reflex executes locally": "System 1 executes locally",
    "Reflex **halts fail-closed**": "System 1 **halts fail-closed**",
    "Reflex replaces this with": "System 1 replaces this with",
    "Reflex ships with": "System 1 ships with",
    "Reflex includes": "System 1 includes",
    "Reflex provides": "System 1 provides",
    "Reflex evaluates": "System 1 evaluates",
    "Reflex evaluated": "System 1 evaluated",
    "Reflex executes": "System 1 executes",
    "Reflex supports": "System 1 supports",
    "Reflex Score": "System 1 Score",
    "Reflex vs cloud API": "System 1 vs cloud API",
    "Reflex Architecture": "System 1 Architecture",
    "Reflex Decision Quality": "System 1 Decision Quality",
    "Reflex Latency Benchmark": "System 1 Latency Benchmark",
    "Reflex Zero Data Egress": "System 1 Zero Data Egress",
    "from reflex": "from system1",
    "import reflex": "import system1",
    "reflex serve": "system1 serve",
    "reflex decide": "system1 decide",
    "reflex bench": "system1 bench",
    "reflex verify-receipt": "system1 verify-receipt",
    "reflex calibrate": "system1 calibrate",
    "reflex compile": "system1 compile",
    "reflex.integrations": "system1.integrations",
    "reflex.compat": "system1.compat",
    "Reflex MCPProxy": "System1 MCPProxy",
    "ReflexMCPProxy": "System1MCPProxy",
    "ReflexGuard": "SystemOneGuard",
    "ReflexEngine": "System1Engine",
    "ReflexGuardCallbackHandler": "System1GuardCallbackHandler",
    "ReflexMetricsExporter": "System1MetricsExporter",
}

for old, new in replacements.items():
    text = text.replace(old, new)

with open("README.md", "w") as f:
    f.write(text)

print("README replacements done.")
