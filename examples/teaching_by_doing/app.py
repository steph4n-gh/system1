"""Capture filing actions, teach one existing System1 head, and preview new files.

The workspace contains text samples only. No real files are moved or opened.
Only explicit human choices or the separately labelled sample session are lessons;
model predictions never become teaching labels automatically.
"""
from collections import Counter
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import shutil
import statistics
import time

from system1 import ChoiceField, DecisionSchema, System1Engine, TeachingSession
from system1.compiler import CompiledSystemOneModel

ROOT = Path(__file__).parent
FOLDERS = ('Finance', 'People', 'Projects')


class Filing(DecisionSchema):
    folder = ChoiceField(options=list(FOLDERS))


def fingerprint(text):
    return sha256(' '.join(text.casefold().split()).encode()).hexdigest()


def observation(document):
    # Only the text visible on the document card. No answer, ID or split tokens.
    return document['title'] + '\n' + document['excerpt']


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


class Session:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.catalog = json.loads((ROOT/'documents.json').read_text(encoding='utf-8'))
        self.documents = {d['id']: d for d in self.catalog['documents']}
        self.records = {}
        self.batch = None
        path = self.directory/'actions.json'
        if path.exists():
            saved = json.loads(path.read_text(encoding='utf-8'))
            if saved.get('version') != 1:
                raise ValueError('Unsupported demonstration file version')
            for row in saved['actions']:
                self.validate_record(row)
                if row['id'] in self.records:
                    raise ValueError('Repeated demonstration id')
                self.records[row['id']] = row
        existing_session = (self.directory/'session.json').exists()
        self.lifecycle = TeachingSession(self.directory, Filing)
        self.sync_lessons(check_existing=existing_session)
        # Earlier versions saved an automatically activated skill under this name.
        # Preserve it only when it has exactly the same filing schema.
        legacy = self.directory/'filing.s1m'
        if legacy.exists() and not self.lifecycle.current_path.exists():
            model = CompiledSystemOneModel.load(legacy)
            if model.schema.schema_digest() != Filing().schema_digest():
                raise ValueError('The saved skill does not use the filing schema')
            shutil.copyfile(legacy, self.lifecycle.current_path)

    @property
    def engine(self):
        return self.lifecycle.engine

    @property
    def teaching(self):
        if self.engine is None:
            return None
        counts = self.engine.model.metadata['sample_counts']['folder']
        return {'fit': counts['fit'], 'calibration': counts['calibration'],
                'conformal_checks': counts['calibration'] - counts['temperature']}

    def sync_lessons(self, *, check_existing=False):
        rows = [{'input': row['input'], 'label': row['folder'],
                 'split': 'teach' if row['purpose'] == 'teach' else 'calibrate',
                 'source': 'human' if row['source'] == 'manual' else 'sample'}
                for row in self.records.values()]
        # Fixed authored answers are recurring development checks, never fitting
        # or calibration examples. Challenge requests have no in-schema label.
        rows += [{'input': observation(doc), 'label': doc['folder'],
                  'split': 'evaluate', 'source': 'sample'}
                 for doc in self.documents.values() if doc['purpose'] == 'evaluate']
        if check_existing:
            desired = {fingerprint(row['input']): row for row in rows}
            existing = {fingerprint(row['input']): row for row in self.lifecycle.records}
            if existing != desired:
                raise ValueError('The demo actions and teaching session differ. Use a separate directory for CLI/Python lessons; no records were overwritten.')
        else:
            self.lifecycle.record_many(rows)

    @staticmethod
    def load_engine(model):
        model.use_cache = False
        return System1Engine(model.schema, model=model, strict_mode=True, use_cache=False)

    def digest(self):
        return sha256(json.dumps(sorted(self.records.values(), key=lambda r:r['id']), sort_keys=True).encode()).hexdigest()

    @staticmethod
    def validate_text(title, excerpt):
        if not isinstance(title, str) or not isinstance(excerpt, str) or not title.strip() or not excerpt.strip() or len(title) > 160 or len(excerpt) > 4000:
            raise ValueError('Use a title up to 160 characters and text up to 4,000 characters')

    def validate_record(self, row):
        if set(row) != {'id','title','excerpt','input','folder','purpose','source'}:
            raise ValueError('Invalid demonstration fields')
        if row['folder'] not in FOLDERS or row['purpose'] not in ('teach','check') or row['source'] not in ('manual','sample'):
            raise ValueError('Invalid folder, purpose or source')
        self.validate_text(row['title'], row['excerpt'])
        if row['input'] != observation(row):
            raise ValueError('Observation must match the visible document')
        if row['id'] in self.documents:
            doc = self.documents[row['id']]
            if row['purpose'] != doc['purpose'] or row['input'] != observation(doc):
                raise ValueError('A reserved document cannot become a teaching example')
        elif row['id'] == 'own-' + fingerprint(row['input']):
            if row['purpose'] != 'teach' or row['source'] != 'manual':
                raise ValueError('Your own text needs an explicit human teaching label')
            if fingerprint(row['input']) in {fingerprint(observation(d)) for d in self.documents.values()}:
                raise ValueError('Sample documents are reserved for their declared purpose')
        else:
            raise ValueError('Unknown demonstration document')

    def save(self):
        self.sync_lessons()
        atomic_json(self.directory/'actions.json', {'version':1, 'actions':list(self.records.values())})
        self.batch = None

    def record_text(self, title, excerpt, folder):
        self.validate_text(title, excerpt)
        text = observation({'title': title, 'excerpt': excerpt})
        row = {'id': 'own-' + fingerprint(text), 'title': title, 'excerpt': excerpt,
               'input': text, 'folder': folder, 'purpose': 'teach', 'source': 'manual'}
        self.validate_record(row)
        self.records[row['id']] = row
        self.save()
        return self.snapshot()

    def record(self, document_id, folder, *, source='manual'):
        if (document_id not in self.documents and document_id not in self.records) or folder not in FOLDERS:
            raise ValueError('Choose a recorded document and one of the three folders')
        doc = self.documents.get(document_id) or self.records[document_id]
        row = {k:doc[k] for k in ('id','title','excerpt','purpose')}
        row.update(input=observation(doc), folder=folder, source=source)
        self.validate_record(row)
        self.records[document_id] = row
        self.save()
        return self.snapshot()

    def remove(self, document_id):
        if document_id not in self.records:
            raise ValueError('That document has no recorded choice')
        self.lifecycle.remove(self.records[document_id]['input'])
        del self.records[document_id]
        self.save()
        return self.snapshot()

    def sample_session(self):
        # Explicitly requested authored examples; preserve every user correction.
        for doc in self.documents.values():
            if doc['purpose'] in ('teach','check') and doc['id'] not in self.records:
                row = {k:doc[k] for k in ('id','title','excerpt','purpose','folder')}
                row.update(input=observation(doc), source='sample')
                self.validate_record(row)
                self.records[doc['id']] = row
        self.save()
        return self.snapshot()

    def teach(self):
        fitting = [r for r in self.records.values() if r['purpose'] == 'teach']
        if {r['folder'] for r in fitting} != set(FOLDERS):
            raise ValueError('Show at least one document in each folder before teaching')
        self.lifecycle.assess()
        return self.snapshot()

    def adopt(self):
        self.lifecycle.adopt()
        self.batch = None
        return self.snapshot()

    def predict(self, title, excerpt):
        self.validate_text(title, excerpt)
        if self.engine is None:
            raise ValueError('Review and adopt a passing candidate first')
        started = time.perf_counter()
        result = self.lifecycle.predict(observation({'title':title,'excerpt':excerpt}))
        return {'folder':result.values['folder'], 'review':bool(result.is_ambiguous),
                'possibilities':result.conformal_sets['folder'], 'milliseconds':(time.perf_counter()-started)*1000}

    def preview(self):
        rows = []
        for doc in self.documents.values():
            if doc['purpose'] not in ('evaluate','challenge'):
                continue
            decision = self.predict(doc['title'],doc['excerpt'])
            rows.append({**{k:doc[k] for k in ('id','title','excerpt','purpose')}, **decision,
                         'destination':'Review' if decision['review'] else decision['folder']})
        self.batch = rows
        # No calls to record(): self-generated answers are never demonstrations.
        return {'rows':rows, 'filed':sum(not r['review'] for r in rows),
                'review':sum(r['review'] for r in rows), 'total':len(rows)}

    def evaluate(self):
        batch = self.preview()
        regular = [r for r in batch['rows'] if r['purpose']=='evaluate']
        challenges = [r for r in batch['rows'] if r['purpose']=='challenge']
        for row in regular:
            row['expected'] = self.documents[row['id']]['folder']
            row['correct'] = row['folder'] == row['expected']
        for row in challenges:
            row['expected_review'] = True
        accepted = [r for r in regular if not r['review']]
        report = {'provenance':self.catalog['provenance'], 'policy':self.catalog['policy'],
                  'actions_sha256':self.digest(), 'model_sha256':sha256(self.lifecycle.current_path.read_bytes()).hexdigest(),
                  'evidence_scope':'Recurring authored development checks; not a fresh independent test.',
                  'teaching':self.teaching, 'skill_bytes':self.lifecycle.current_path.stat().st_size,
                  'cases':len(regular),'correct':sum(r['correct'] for r in regular),
                  'accepted':len(accepted),'accepted_correct':sum(r['correct'] for r in accepted),
                  'challenge_cases':len(challenges),'challenges_sent_to_review':sum(r['review'] for r in challenges),
                  'median_decision_ms':statistics.median(r['milliseconds'] for r in batch['rows']),
                  'teacher_calls_during_prediction':0, 'rows':batch['rows'],
                  'limitations':['Authored fictional previews, not a representative document benchmark.',
                                 'Expected folders implement the displayed sample policy, not arbitrary personal preferences.',
                                 'Calibration labels calibrate review behavior; the sample batch is reused for candidate development checks.',
                                 'A confident prediction can still be wrong; no production readiness claim.',
                                 'No PDF extraction, OCR, filesystem actions or automatic desktop observation.']}
        atomic_json(self.directory/'preview-report.json', report)
        return report

    def export_lessons(self):
        return {name:{'folder':[(r['input'],r['folder']) for r in self.records.values() if r['purpose']==purpose]}
                for name,purpose in [('exemplars','teach'),('calibration_exemplars','check')]}

    def snapshot(self):
        documents = [{k:v for k,v in d.items() if k != 'folder'} for d in self.documents.values()]
        documents += [{k:r[k] for k in ('id', 'title', 'excerpt', 'purpose')}
                      for r in self.records.values() if r['id'].startswith('own-')]
        lifecycle = self.lifecycle.snapshot()
        candidate = lifecycle['report']
        if candidate is not None:
            candidate = {**candidate, 'stale': lifecycle['candidate_stale'],
                         'adopted': lifecycle['adopted']}
        return {'policy':self.catalog['policy'], 'documents':documents,
                'actions':list(self.records.values()), 'counts':dict(Counter(r['purpose'] for r in self.records.values())),
                'sources':dict(Counter(r['source'] for r in self.records.values())), 'taught':self.engine is not None,
                'teaching':self.teaching, 'folders':FOLDERS, 'candidate':candidate,
                'development_checks':sum(d['purpose'] == 'evaluate' for d in self.documents.values()),
                'evidence_scope':'The same 30 authored filing samples are recurring development checks, not a fresh test.'}


def serve(directory, port=8791):
    session = Session(directory)
    html = (ROOT/'index.html').read_bytes()
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, value, kind='application/json', filename=None):
            body = json.dumps(value).encode() if kind=='application/json' else value
            self.send_response(status)
            self.send_header('Content-Type',kind)
            self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            if filename:
                self.send_header('Content-Disposition',f'attachment; filename="{filename}"')
            self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            if self.path=='/': return self.reply(200,html,'text/html; charset=utf-8')
            if self.path=='/api/state': return self.reply(200,session.snapshot())
            if self.path=='/api/lessons': return self.reply(200,session.export_lessons(),filename='filing-lessons.json')
            if self.path=='/api/actions': return self.reply(200,{'version':1,'actions':list(session.records.values())},filename='filing-actions.json')
            if self.path=='/api/skill' and session.engine is not None:
                return self.reply(200,session.lifecycle.current_path.read_bytes(),'application/octet-stream','filing.s1m')
            return self.reply(404,{'error':'Not found; adopt a passing candidate before downloading the skill'})

        def do_POST(self):
            host, origin = self.headers.get('Host',''),self.headers.get('Origin')
            if host not in (f'127.0.0.1:{port}',f'localhost:{port}') or (origin and origin!=f'http://{host}'):
                return self.reply(403,{'error':'Use the local demonstration page'})
            try:
                length = int(self.headers.get('Content-Length','0'))
                if not 0<length<=65536 or self.headers.get_content_type()!='application/json':
                    raise ValueError('Send a small JSON object')
                data = json.loads(self.rfile.read(length))
                if not isinstance(data,dict): raise ValueError('Expected an object')
                if self.path=='/api/record' and set(data)=={'id','folder'}: result=session.record(data['id'],data['folder'])
                elif self.path=='/api/record-text' and set(data)=={'title','excerpt','folder'}: result=session.record_text(**data)
                elif self.path=='/api/remove' and set(data)=={'id'}: result=session.remove(data['id'])
                elif self.path=='/api/sample' and not data: result=session.sample_session()
                elif self.path=='/api/teach' and not data: result=session.teach()
                elif self.path=='/api/adopt' and not data: result=session.adopt()
                elif self.path=='/api/preview' and not data: result=session.preview()
                elif self.path=='/api/evaluate' and not data: result=session.evaluate()
                elif self.path=='/api/predict' and set(data)=={'title','excerpt'}: result=session.predict(**data)
                else: return self.reply(404,{'error':'Unknown demonstration action'})
                return self.reply(200,result)
            except (ValueError,TypeError,KeyError) as error:
                return self.reply(400,{'error':str(error)})

        def log_message(self,*args): pass
    print(f'Teach by doing: http://127.0.0.1:{port}',flush=True)
    HTTPServer(('127.0.0.1',port),Handler).serve_forever()
