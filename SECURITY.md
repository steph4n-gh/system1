# Security policy

## Reporting a vulnerability

Report vulnerabilities privately through [GitHub's private vulnerability reporting form](https://github.com/steph4n-gh/system1/security/advisories/new). If private reporting is unavailable, open an issue requesting a private contact without posting exploit details, credentials, or sensitive data.

Include the affected version, configuration, expected boundary, reproduction steps using harmless sentinel tools, and observed result. Please avoid testing against services or data you do not own.

## Supported scope

System 1 is in beta. Fixes target the latest release; older versions are not maintained as separate branches. Report failures in deterministic policy enforcement, signed receipt verification, ledger integrity, model loading, or integration dispatch.

The statistical classifier is not a general-purpose security detector. The application supplies authenticated identity, isolates tools, protects policy and signing keys, and enforces returned decisions. Read [deployment boundaries](docs/deployment.md) before using the library for consequential actions.
