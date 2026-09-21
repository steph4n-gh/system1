import hashlib, json, sys
import numpy as np
from threadpoolctl import threadpool_limits
sys.path.insert(0, 'benchmarks/quality/n8n_gauntlet')
from develop import load_splits, OUTPUT
from encoder_probe import Encoder, PreparedFeatures
from boundary_development import compile_head
from runtime_probe import LocalCandidate
from system1 import ChoiceField, DecisionSchema, SystemOneCompiler
from teach_boundaries import HERE
with threadpool_limits(limits=1):
    data = load_splits('banking77')
    original = data['fit']
    extra = json.loads((HERE / 'results/contrast-lessons.json').read_text())['lessons']
    augmented = original + extra
    parent = LocalCandidate(HERE / 'artifacts/banking-review')
    encoder = Encoder('bge-small')
    vectors, caches = {}, {}
    for part, rows in dict(augmented=augmented, calibration=data['calibration']).items():
        key = hashlib.sha256((encoder.projector_digest() + json.dumps(rows, sort_keys=True)).encode()).hexdigest()
        path = OUTPUT / f'features-{key}.npy'
        vectors[part] = np.load(path, allow_pickle=False)
        caches[part] = dict(path=str(path.relative_to(HERE.parents[2])), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    labels = parent.model.heads['intent'].options
    y = np.asarray([labels.index(r['label']) for r in augmented])
    np.testing.assert_array_equal(parent.prototypes, vectors['augmented'][np.argsort(y, kind='stable')])
    schema = type('Intent', (DecisionSchema,), {'intent': ChoiceField(options=labels)})
    projector = PreparedFeatures(augmented + data['calibration'], np.concatenate([vectors['augmented'], vectors['calibration']]), encoder)
    compiler = SystemOneCompiler(schema, projector=projector, choice_solver='logistic', regularization=.01)
    rebuilt = compile_head(compiler, augmented, data['calibration']).heads['intent']
    published = parent.model.heads['intent']
    report = dict(scope='Banking unchanged-control reconstruction diagnostic, no corrected labels or new runtime timing', caches=caches,
        prototypes_exact=True, max_weight_difference=float(np.max(np.abs(rebuilt.weights-published.weights))),
        max_bias_difference=float(np.max(np.abs(rebuilt.biases-published.biases))), temperature_difference=float(abs(rebuilt.temperature-published.temperature)))
    print(json.dumps(report, indent=2))
