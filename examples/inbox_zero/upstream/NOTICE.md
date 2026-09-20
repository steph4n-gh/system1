# Inbox Zero patch provenance

`system1.patch` modifies Inbox Zero at upstream revision
`2571ce4970aa5d024bb664a6739bd91b1172c367`:
https://github.com/elie222/inbox-zero/tree/2571ce4970aa5d024bb664a6739bd91b1172c367

The patch and upstream-derived material in this directory are supplied under the
accompanying Inbox Zero `LICENSE` (AGPLv3 with the upstream additional terms),
not under System1's Apache-2.0 license. Upstream copyright notices remain intact.
Modifications authored by steph4n-gh on 2026-09-20 add the System1 provider,
configuration, explicit review handling, and regression tests. Applying the patch
to the pinned source gives the complete modified application source.

The standalone Python integration and independently authored teaching examples
outside this directory remain part of System1. The exporter obtains upstream
rule definitions and evaluation fixtures into the user's ignored output directory
and includes the upstream license there. No private mailbox data is distributed.

This is a proposed integration; it is not endorsed by or merged into Inbox Zero.
