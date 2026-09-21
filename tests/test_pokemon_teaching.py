"""Behavioral and evidence checks for the two-skill teaching experiment."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import random
import shutil
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from examples.gaming.pokemon_teaching import battle as b
from examples.gaming.pokemon_teaching.server import Lab
from examples.gaming.pokemon_teaching.red import read_state
from examples.gaming.skill_playground.quality import objective_lessons, stop_on_review
from examples.gaming.skill_playground.experiment import lessons as courier_lessons, run_episode
from examples.gaming.skill_playground.world import World


@pytest.fixture(scope='module')
def taught(tmp_path_factory):
    root=tmp_path_factory.mktemp('pokemon-teaching')
    b.prepare(root)
    return root


def test_disjoint_effective_inputs_and_matched_budgets(taught):
    data=json.loads((taught/'lessons.json').read_text())
    for name,rows in data.items():
        fit={r['input'] for r in rows if r['split']=='teach'}
        cal={r['input'] for r in rows if r['split']=='calibration'}
        assert fit and len(cal)>=19 and not fit & cal
        assert len(fit)+len(cal)==len(rows)
    assert Counter(r['split'] for n in ('move','healing') for r in data[n])==Counter(r['split'] for r in data['flat'])
    assert Counter(r['split'] for r in data['flat'])==Counter(r['split'] for r in data['flat-corrected'])
    assert Counter(a['split'] for a,z in zip(data['healing'],data['healing-corrected']) if a['label']!=z['label'])=={'teach':21,'calibration':6}


def test_saved_skills_fix_a_battle_without_teachers(taught,tmp_path,monkeypatch):
    for name in ('move','healing','healing-corrected'):
        shutil.copyfile(taught/f'{name}.s1m',tmp_path/f'{name}.s1m')
    def forbidden(*args,**kwargs):
        raise AssertionError('Inference called a teacher')
    monkeypatch.setattr(b,'rule_decision',forbidden)
    monkeypatch.setattr(b,'healing_label',forbidden)
    original=b.play(tmp_path,8004,'survival')
    fixed=b.play(tmp_path,8004,'survival',corrected=True,strict=True)
    assert original['outcome']=='DEFEAT'
    assert fixed['outcome']=='VICTORY' and fixed['review_steps']==0
    assert fixed['model_calls']==3*fixed['turns'] and fixed['teacher_calls']==0
    assert fixed['events'][0]['action']=='use_item'


def test_live_correction_preserves_move_object_file_and_predictions(taught):
    lab=Lab(taught)
    engine=lab.agents['network'].move
    digest=sha256((taught/'move.s1m').read_bytes()).hexdigest()
    rows=json.loads((taught/'lessons.json').read_text())['move']
    before=[engine.decide(r['input'],record_receipt=False).to_dict() for r in rows[:30]]
    lab.correct(True)
    assert lab.agents['network'].move is engine
    assert sha256((taught/'move.s1m').read_bytes()).hexdigest()==digest
    after=[engine.decide(r['input'],record_receipt=False).to_dict() for r in rows[:30]]
    for a,z in zip(before,after):
        for key in ('values','probabilities','conformal_sets','is_ambiguous'):
            assert a[key]==z[key]
    while not all(x.done for x in lab.battles.values()):lab.step()
    assert lab.snapshot()['lanes']['network']['outcome']=='VICTORY'
    lab.correct(False)
    while not all(x.done for x in lab.battles.values()):lab.step()
    assert lab.snapshot()['lanes']['network']['outcome']=='DEFEAT'
    for bad in ({'seed':True},{'family':'unknown'},{'strict':'no'},{'command':'x'},[]):
        with pytest.raises(ValueError):lab.reset(bad)


def test_review_stops_before_physics_and_rng_isolated(taught,monkeypatch):
    agent=b.Agent(taught,stop_on_review=True)
    monkeypatch.setattr(agent,'decide',lambda state:{'action':'fight','chosen_move':'move_slot_1','review':True,'calls':3})
    state=b.encounter(8004,'survival')
    original=asdict(state)
    battle=b.Battle(state,agent,8004)
    rng=random.getstate()
    battle.step()
    assert battle.stopped and asdict(state)==original and random.getstate()==rng
    assert battle.result()['outcome']=='REVIEW'


def test_courier_targeted_labels_and_strict_stop():
    old=courier_lessons()['objective']
    new=objective_lessons(old)
    assert all(r in new for r in old)
    fit={r['input'] for r in new if r['split']=='teach'}
    cal={r['input'] for r in new if r['split']=='calibration'}
    assert not fit & cal
    assert Counter(r['split'] for r in new)-Counter(r['split'] for r in old)=={'teach':104,'calibration':45}
    controller=stop_on_review(lambda world:{'action':'east','review':True,'calls':9})
    row=run_episode(controller,6000)
    assert row['steps']==0 and row['paused_for_review'] and not row['success']


def test_red_reader_uses_actual_stats_moves_and_bag_without_defaults():
    memory=bytearray(65536)
    rom=bytearray(0x40000)
    for base,species,typ,hp in ((0xd014,177,21,17),(0xcfe5,153,22,13)):
        memory[base]=species
        memory[base+2]=hp
        memory[base+5]=memory[base+6]=typ
        memory[base+8]=33
        memory[base+14]=5
        memory[base+16]=20
        for offset,n in ((17,11),(19,12),(21,10),(23,13)):
            memory[base+offset+1]=n
        memory[base+25]=0xc0|7
    memory[0xd057]=2
    start=0x38000+32*6
    rom[start:start+6]=bytes([33,0,35,0,242,35])
    state=read_state(memory,rom)
    assert state.player_pokemon.name=='Squirtle'
    assert state.opponent_pokemon.max_hp==20 and state.opponent_pokemon.current_hp==13
    assert state.player_pokemon.attack==11 and state.player_pokemon.special==13
    assert len(state.player_pokemon.moves)==1 and state.player_pokemon.moves[0].pp==7
    assert state.player_pokemon.moves[0].category=='physical'
    assert state.inventory=={} and state.party==[] and not state.can_run
    memory[0xd31d]=1;memory[0xd31e]=0x14;memory[0xd31f]=2
    assert read_state(memory,rom).inventory=={'Potion':2}
    memory[0xd057]=0
    with pytest.raises(ValueError):read_state(memory,rom)


def test_published_battle_evidence_replays_all_episodes():
    import zipfile
    root=Path(__file__).resolve().parents[1]/'examples/gaming/pokemon_teaching'
    with zipfile.ZipFile(root/'results/battle-evidence.zip') as archive:
        manifest=json.loads(archive.read('manifest.json'))
        for name,expected in manifest.items():
            assert sha256(archive.read(name)).hexdigest()==expected
        for name,source in json.loads(archive.read('sources.json')).items():
            path=root/name if (root/name).exists() else root.parent/name
            # This archive preserves the original GUI/ROM implementation. Only
            # the simulator and policy still need to match for action replay.
            if name in ('battle.py', 'pokemon_battle_system1.py'):
                assert path.read_text()==source
        report=json.loads(archive.read('report.json'))
        data=json.loads(archive.read('lessons.json'))
        for name,rows in data.items():
            assert b.digest(rows)==report['teaching'][name]['lessons_sha256']
        assert json.loads((root/'results/summary.json').read_text())=={k:v for k,v in report.items() if k!='episodes'}
        class Recorded:
            def __init__(self,events,strict):
                self.recorded,self.strict=events,strict
                self.events=[]
                self.escalations=0
            def evaluate(self,state):
                event=deepcopy(self.recorded[len(self.events)])
                assert event['facts']=={'hp':state.player_pokemon.current_hp,'opponent_hp':state.opponent_pokemon.current_hp}
                self.events.append(event)
                if event['review'] and self.strict:
                    raise b.ReviewStop()
                return {**event,'is_ambiguous':event['review']},False,''
        count=0
        for family,controllers in report['episodes'].items():
            for name,rows in controllers.items():
                for row in rows:
                    state=b.encounter(row['seed'],family)
                    assert b.digest(asdict(state))==b.digest(row['initial'])
                    battle=b.Battle(state,Recorded(row['events'],name.endswith('_strict')),row['seed'])
                    while not battle.done:battle.step()
                    replay=battle.result()
                    assert b.digest(replay)==b.digest(row)
                    count+=1
                assert b.summarize(rows)==report['results'][family][name]
        assert count==1680
        rom=json.loads(archive.read('rom-report.json'))
        assert rom['outcome']=='VICTORY' and rom['turns']==5 and rom['review_steps']==0
        for name,expected in rom['skill_sha256'].items():
            assert sha256(archive.read(name)).hexdigest()==expected
        assert rom['initial']['player_pokemon']['name']=='Squirtle'
        assert rom['initial']['opponent_pokemon']['name']=='Bulbasaur'
        assert not any(name.endswith(('.gb','.gbc','.state','.ram','.png')) for name in archive.namelist())


def test_courier_quality_evidence_replays_strict_stops():
    import zipfile
    from examples.gaming.skill_playground.experiment import summarize
    root=Path(__file__).resolve().parents[1]/'examples/gaming/skill_playground'
    with zipfile.ZipFile(root/'results/quality-evidence.zip') as archive:
        for name,expected in json.loads(archive.read('manifest.json')).items():
            assert sha256(archive.read(name)).hexdigest()==expected
        assert archive.read('quality.py').decode()==(root/'quality.py').read_text()
        report=json.loads(archive.read('quality-report.json'))
        for suite,controllers in report['episodes'].items():
            for name,rows in controllers.items():
                for row in rows:
                    world=World(row['seed'],purple=suite=='changed_terrain',purple_dangerous=suite=='changed_terrain')
                    for action,review in zip(row['actions'],row['review_flags']):
                        if name.endswith('stop_on_review') and review:
                            assert action is None
                        if action is None:world.paused_for_review=True
                        else:world.step(action)
                    assert (world.success,world.alive,world.ticks)==(row['success'],row['alive'],row['steps'])
                assert summarize(rows)==report['results'][suite][name]
        assert report['results']['unseen']['before_stop_on_review']['successes']==22
        assert report['results']['unseen']['after_stop_on_review']['successes']==31
