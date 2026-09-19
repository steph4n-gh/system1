import re

with open("README.md", "r") as f:
    text = f.read()

# Replace main references
replacements = {
    "Why System 1": "Why System 1",
    "Without System 1": "Without System 1",
    "With System 1": "With System 1",
    "System 1 retains": "System 1 retains",
    "System 1 is the on-metal": "System 1 is the on-metal",
    "System 1 executes locally": "System 1 executes locally",
    "System 1 **halts fail-closed**": "System 1 **halts fail-closed**",
    "System 1 replaces this with": "System 1 replaces this with",
    "System 1 ships with": "System 1 ships with",
    "System 1 includes": "System 1 includes",
    "System 1 provides": "System 1 provides",
    "System 1 evaluates": "System 1 evaluates",
    "System 1 evaluated": "System 1 evaluated",
    "System 1 executes": "System 1 executes",
    "System 1 supports": "System 1 supports",
    "System 1 Score": "System 1 Score",
    "System 1 vs cloud API": "System 1 vs cloud API",
    "System 1 Architecture": "System 1 Architecture",
    "System 1 Decision Quality": "System 1 Decision Quality",
    "System 1 Latency Benchmark": "System 1 Latency Benchmark",
    "System 1 Zero Data Egress": "System 1 Zero Data Egress",
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
    "System 1 MCPProxy": "System1 MCPProxy",
    "SystemOneMCPProxy": "System1MCPProxy",
    "SystemOneGuard": "SystemOneGuard",
    "SystemOneEngine": "System1Engine",
    "SystemOneGuardCallbackHandler": "System1GuardCallbackHandler",
    "SystemOneMetricsExporter": "System1MetricsExporter",
}

for old, new in replacements.items():
    text = text.replace(old, new)

with open("README.md", "w") as f:
    f.write(text)

print("README replacements done.")
