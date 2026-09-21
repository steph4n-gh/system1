"""A local GUI for live teaching and freshly executed battle decisions."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import statistics
import time

from .battle import Agent, Battle, encounter, corrected_rows, teach, load


class Lab:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.agents = {name: Agent(directory,mode=mode) for name,mode in
                       (('network','network'),('single','single'),('rules','rules'))}
        self.corrected = False
        self.teaching = None
        self.reset({})

    def reset(self, settings):
        if not isinstance(settings,dict) or set(settings) - {'seed','family','strict'}:
            raise ValueError('Unknown settings')
        seed, family, strict = settings.get('seed',8004),settings.get('family','survival'),settings.get('strict',False)
        if type(seed) is not int or not 0<=seed<=1000000:
            raise ValueError('Seed must be an integer from 0 to 1000000')
        if family not in ('healthy','pressure','mixed','survival') or type(strict) is not bool:
            raise ValueError('Invalid battle settings')
        self.settings = {'seed':seed,'family':family,'strict':strict}
        state = encounter(seed,family)
        self.battles = {}
        for name,agent in self.agents.items():
            agent.events = []
            agent.total_decisions = agent.escalations = 0
            agent.total_latency_ms = 0
            agent.stop_on_review = strict
            self.battles[name] = Battle(deepcopy(state),agent,seed)
        return self.snapshot()

    def correct(self, enabled):
        if type(enabled) is not bool:
            raise ValueError('Correction must be a boolean')
        if enabled:
            original = json.loads((self.directory/'lessons.json').read_text())['healing']
            rows = corrected_rows(original)
            changes = Counter(r['split'] for old,r in zip(original,rows) if old['label']!=r['label'])
            started = time.perf_counter()
            self.agents['network'].healing = teach('healing',rows,self.directory/'healing-corrected.s1m')
            self.teaching = {'milliseconds':(time.perf_counter()-started)*1000,'changed_labels':dict(changes)}
        else:
            self.agents['network'].healing = load(self.directory/'healing.s1m')
            self.teaching = None
        self.corrected = enabled
        return self.reset(self.settings)

    def step(self):
        for battle in self.battles.values():
            battle.step()
        return self.snapshot()

    def snapshot(self):
        lanes = {}
        for name,b in self.battles.items():
            events = b.agent.events
            lanes[name] = {'player':asdict(b.state.player_pokemon), 'opponent':asdict(b.state.opponent_pokemon),
                           'inventory':b.state.inventory, 'done':b.done,
                           'outcome':('REVIEW' if b.stopped else b.state.outcome or ('TIMEOUT' if b.done else 'READY')),
                           'turns':len(events),'review_steps':sum(e['review'] for e in events),
                           'potions':b.potions_before-sum(b.state.inventory.values()),
                           'decision':events[-1] if events else None,
                           'median_ms':statistics.median(e['latency_ms'] for e in events) if events else 0,
                           'log':b.state.battle_log[-4:]}
        return {'settings':self.settings,'corrected':self.corrected,'teaching':self.teaching,'lanes':lanes}


def serve(directory, port=8790, rom_path=None):
    root = Path(directory)
    lab = Lab(root)
    rom = None
    html = Path(__file__).with_name('index.html').read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, content, kind='application/json'):
            body = json.dumps(content).encode() if kind=='application/json' else content
            self.send_response(status)
            self.send_header('Content-Type',kind)
            self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path=='/':
                return self.reply(200,html,'text/html; charset=utf-8')
            if self.path=='/api/state':
                return self.reply(200,lab.snapshot())
            if self.path=='/api/report':
                report=json.loads((root/'report.json').read_text())
                return self.reply(200,{k:v for k,v in report.items() if k!='episodes'})
            if self.path=='/api/lessons':
                return self.reply(200,json.loads((root/'lessons.json').read_text()))
            if self.path=='/api/rom/state':
                return self.reply(200,rom.snapshot() if rom else {'available':rom_path is not None,'started':False})
            return self.reply(404,{'error':'Not found'})

        def do_POST(self):
            nonlocal rom
            host=self.headers.get('Host','')
            origin=self.headers.get('Origin')
            if host not in (f'127.0.0.1:{port}',f'localhost:{port}') or (origin and origin!=f'http://{host}'):
                return self.reply(403,{'error':'Use the local teaching page'})
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=4096 or self.headers.get_content_type()!='application/json':
                    raise ValueError('Send a small JSON object')
                data=json.loads(self.rfile.read(length))
                if not isinstance(data,dict):
                    raise ValueError('Expected a JSON object')
                if self.path=='/api/reset':
                    result=lab.reset(data)
                elif self.path=='/api/step' and not data:
                    result=lab.step()
                elif self.path=='/api/teach' and set(data)=={'enabled'}:
                    result=lab.correct(data['enabled'])
                elif self.path=='/api/rom/start' and not data and rom_path is not None:
                    from .red import RomBattle
                    if rom:
                        rom.close()
                        rom=None
                    rom=RomBattle(rom_path,root,corrected=lab.corrected)
                    result=rom.snapshot()
                elif self.path=='/api/rom/step' and not data and rom:
                    result=rom.step()
                    if rom.done:
                        (root/'rom-live-report.json').write_text(json.dumps(rom.report(),indent=2)+'\n')
                else:
                    return self.reply(404,{'error':'Unknown action'})
                return self.reply(200,result)
            except (ValueError,TypeError,KeyError) as error:
                return self.reply(400,{'error':str(error)})
            except (RuntimeError,ImportError) as error:
                return self.reply(422,{'error':str(error)})

        def log_message(self,*args):
            pass

    print(f'Pokemon teaching lab: http://127.0.0.1:{port}',flush=True)
    try:
        HTTPServer(('127.0.0.1',port),Handler).serve_forever()
    finally:
        if rom:
            rom.close()
