# Banking support data

[banking_support.json](banking_support.json) adapts a fixed three-intent subset of
the BANKING77 dataset by Iñigo Casanueva, Tadas Temcinas, Daniela Gerz,
Matthew Henderson, and Ivan Vulić (2020), described in
[Efficient Intent Detection with Dual Sentence Encoders](https://arxiv.org/abs/2003.04807).
The data is licensed under [CC BY 4.0](BANKING77_LICENSE.md), separately from
System 1's Apache 2.0 code license. Original source:
[PolyAI/task-specific-datasets](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data).

Changes from the original: select `card_arrival`, `lost_or_stolen_card`, and
`cash_withdrawal_charge`; convert CSV to the example's JSON format; add source
row indices and case groups; split the official training rows into teaching and
calibration. Query text and labels are unchanged. Row indices are zero-based,
excluding the CSV header.

These three queues were selected before evaluating any test predictions.
Within each intent, normalize training prompts to lowercase word tokens, keep
the first row per normalized duplicate, and sort groups by the SHA-256 digest
of that normalized text. The first 30% of groups, rounded up, provide calibration;
the rest provide teaching. All 120 official test rows for these intents are retained.
Any training group overlapping an official test group would be removed from
training; there were no such overlaps. The result is 287 teaching, 125 calibration,
and 120 evaluation cases. The compiler further separates temperature fitting and
conformal calibration using its existing prompt-disjoint split.

The official test rows did not inform task selection, examples, model settings,
or thresholds. After the first evaluation, the data and settings were left fixed.
The result describes this bounded task, not all 77 intents, out-of-scope detection,
or live traffic supplied by System 1 users. The release example itself does not compare against Jev. A later [observation experiment](../../benchmarks/quality/workloads/README.md) uses Jev on the fitting/calibration stream and records that automatic promotion did not qualify.

To audit provenance using the pinned upstream revision:

```bash
git clone https://github.com/PolyAI-LDN/task-specific-datasets /tmp/system1-banking-source
git -C /tmp/system1-banking-source checkout 57ec275d8078af65b7731c2a98be812d844a6d6b
python benchmarks/quality/verify_banking_source.py /tmp/system1-banking-source/banking_data
```

Normal teaching and evaluation use the bundled JSON and require no download.
The verification script checks upstream checksums, unchanged text and labels,
the deterministic teaching/calibration partition, and the complete selected test
slice. See the [1.0.1 results](../../docs/releases/1.0.1.md).
