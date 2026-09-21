"""Two saved skills wired into the existing, explicitly simplified battle loop."""
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import random
import platform
import statistics
import time

from system1 import ChoiceField, DecisionSchema, ScoreField, System1Engine
from system1.compiler import SystemOneCompiler, CompiledSystemOneModel
from system1.core.text import TfidfProjector
from ..pokemon_battle_system1 import (
    BattleState, BattleType, Pokemon, PokemonMove, battle_choices,
    estimate_move_damage, get_type_effectiveness, run_battle_simulation,
)


class Move(DecisionSchema):
    preference = ScoreField(min_value=0, max_value=1)


class Healing(DecisionSchema):
    action = ChoiceField(options=['fight', 'use_item'])


class Flat(DecisionSchema):
    action = ChoiceField(options=['use_item'] + [f'move_slot_{i}' for i in range(1, 5)])


SCHEMAS = {'move': Move, 'healing': Healing, 'flat': Flat}


def digest(value):
    return sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load(path):
    model = CompiledSystemOneModel.load(path)
    model.use_cache = False
    return System1Engine(model.schema, model=model, strict_mode=True, use_cache=False)


def teach(name, rows, path):
    field = next(iter(SCHEMAS[name]().fields))
    fitting = [r for r in rows if r['split'] == 'teach']
    calibration = [r for r in rows if r['split'] == 'calibration']
    projector = TfidfProjector.fit([r['input'] for r in fitting], max_features=1024)
    compiler = SystemOneCompiler(SCHEMAS[name], projector=projector, regularization=.01)
    model = compiler.compile({field: [(r['input'], r['label']) for r in fitting]}, augment=False,
                             calibration_exemplars={field: [(r['input'], r['label']) for r in calibration]})
    model.save(path)
    return load(path)


def move_prompt(state, move):
    p, o = state.player_pokemon, state.opponent_pokemon
    attack = p.special if move.category == 'special' else p.attack
    defense = o.special if move.category == 'special' else o.defense
    # Game facts, with numeric buckets. No chosen move or teacher output here.
    ratio = round(attack / max(1, defense), 6)
    return (f'power_{move.power} accuracy_{move.accuracy} level_{p.level // 5 * 5} '
            f'ratio_{ratio} effectiveness_{get_type_effectiveness(move.move_type, o.types)} '
            f'stab_{int(move.move_type in p.types)} status_{int(move.category == "status")}')


def healing_facts(state, move):
    p, o = state.player_pokemon, state.opponent_pokemon
    incoming = max((estimate_move_damage(o, p, m) for m in o.moves if m.is_usable()), default=0)
    outgoing = estimate_move_damage(p, o, move) if move else 0
    amount = 50 if state.inventory.get('Super Potion', 0) else 20 if state.inventory.get('Potion', 0) else 0
    return {'critical': p.hp_ratio < .25,
            'lethal': p.current_hp <= incoming / .925,
            'finish': outgoing * .9 >= o.current_hp,
            'item': amount > 0 and p.current_hp < p.max_hp,
            'restores': min(amount, p.max_hp - p.current_hp) > incoming,
            'hp': p.current_hp, 'enemy_hp': o.current_hp, 'incoming': round(incoming, 2)}


def healing_prompt(values):
    flags = [f'{k}_{int(values[k])}' for k in ('critical','lethal','finish','item','restores')]
    observed = [f'{k}_{values[k]}' for k in ('hp','enemy_hp','incoming')]
    return ' '.join(flags + [a + '__' + b for a, b in combinations(flags, 2)] + observed)


def flat_prompt(state):
    parts = []
    for m in state.player_pokemon.moves:
        parts.extend(f'{m.slot}_{t}' for t in move_prompt(state, m).split())
        parts.extend(f'{m.slot}_{t}' for t in healing_prompt(healing_facts(state, m)).split())
        parts.append(f'{m.slot}_usable_{int(m.is_usable())}')
    return ' '.join(parts)


def healing_label(values, corrected=False):
    if corrected:
        heal = values['item'] and (values['critical'] or (values['lethal'] and not values['finish'] and values['restores']))
    else:
        heal = values['item'] and values['critical']
    return 'use_item' if heal else 'fight'


def rule_decision(state, corrected=False):
    # This reproduces the existing default: immediate damage and HP < 25%.
    _, moves = battle_choices(state)
    move = max(moves, key=lambda m: (estimate_move_damage(state.player_pokemon,state.opponent_pokemon,m), -int(m.slot[-1]))) if moves else None
    return {'action': healing_label(healing_facts(state, move), corrected),
            'chosen_move': move.slot if move else 'struggle', 'review': False, 'calls': 0}


def encounter(seed, family='mixed'):
    rng = random.Random(seed)
    types = ('Normal', 'Water', 'Fire', 'Electric', 'Grass', 'Rock')
    level = rng.choice((20, 25, 30, 35))
    moves = [PokemonMove(f'move_slot_{i+1}', name, typ, power, accuracy, rng.choice((0, 8, 15)), 15)
             for i, (name, typ, power, accuracy) in enumerate(rng.sample([
                 ('Surf','Water',95,100), ('Thunderbolt','Electric',95,100),
                 ('Flamethrower','Fire',95,100), ('Razor Leaf','Grass',55,95),
                 ('Ice Beam','Ice',95,100), ('Tackle','Normal',35,95),
                 ('Bite','Normal',60,100), ('Hydro Pump','Water',120,80)], 4))]
    moves[0].pp = 15
    p = Pokemon('Partner', level, 80, 80, (rng.choice(types),), moves,
                attack=rng.choice((40,60,80)), defense=rng.choice((40,60,80)),
                special=rng.choice((40,60,80)), speed=50)
    o = Pokemon('Opponent', level, rng.choice((70,90,110)), 110, (rng.choice(types),),
                [PokemonMove('move_slot_1','Tackle','Normal',rng.choice((35,50,65)),100,30,30,'physical')],
                attack=rng.choice((50,70,90)), defense=rng.choice((40,60,80)),
                special=rng.choice((40,60,80)), speed=40)
    if family == 'survival':
        # A declared stress family for the 25%-HP rule: modest remaining HP,
        # a strong neutral hit, and healing resources. No outcome filtering.
        p.level = o.level = 25
        p.types = o.types = ('Normal',)
        p.current_hp = rng.randrange(21,36)
        p.attack = p.defense = p.special = 60
        p.moves = [PokemonMove('move_slot_1','Bite','Normal',60,100,15,15,'physical'),
                   PokemonMove('move_slot_2','Tackle','Normal',35,95,15,15,'physical')]
        if rng.randrange(2):
            p.moves.reverse()
            for i,m in enumerate(p.moves):
                m.slot = f'move_slot_{i+1}'
        o.current_hp = o.max_hp = rng.randrange(50,71)
        o.attack = 90
        o.defense = o.special = 60
        o.moves = [PokemonMove('move_slot_1','Tackle','Normal',50,100,30,30,'physical')]
    elif family == 'healthy':
        p.current_hp = rng.randrange(60,81)
    elif family == 'pressure':
        # Broad low-health encounters; not filtered by any controller's outcome.
        p.current_hp = rng.randrange(20,41)
    else:
        p.current_hp = rng.randrange(8,81)
    return BattleState(p, o, BattleType.TRAINER, party=[],
                       inventory={'Super Potion': 2 if family == 'survival' else rng.choice((0,1,2))}, can_run=False)


def lessons(count=1200):
    data = {name: {} for name in SCHEMAS}
    # Reserve effective input groups before fitting; no duplicate leakage into cal.
    def add(name, text, label):
        split = 'calibration' if int(digest(text)[:8], 16) % 4 == 0 else 'teach'
        previous = data[name].get(text)
        if previous:
            # A bucket can contain slightly different continuous damage. Average
            # scores by collecting raw candidate values below, never class labels.
            if name != 'move' and previous['label'] != label:
                raise AssertionError('Conflicting discrete teaching labels')
            previous['_scores'].append(label)
        elif previous is None:
            data[name][text] = {'input': text, 'label': label, 'split': split, '_scores': [label]}
    for seed in range(count):
        state = encounter(seed)
        for move in state.player_pokemon.moves:
            damage = estimate_move_damage(state.player_pokemon, state.opponent_pokemon, move)
            add('move', move_prompt(state, move), damage / (damage + 20))
            # Include non-damaging move lessons for the real starter's Tail Whip.
            status = PokemonMove(move.slot, 'Tail Whip', 'Normal', 0, 100, 30, 30, 'status')
            add('move', move_prompt(state, status), 0.0)
            values = healing_facts(state, move)
            add('healing', healing_prompt(values), healing_label(values))
        decision = rule_decision(state)
        text = flat_prompt(state)
        add('flat', text, 'use_item' if decision['action'] == 'use_item' else decision['chosen_move'])
        revised = rule_decision(state, True)
        data['flat'][text]['corrected_label'] = 'use_item' if revised['action']=='use_item' else revised['chosen_move']
    result = {}
    for name, mapping in data.items():
        result[name] = []
        for r in mapping.values():
            scores = r.pop('_scores')
            if name == 'move':
                r['label'] = statistics.mean(scores)
            result[name].append(r)
    budgets = Counter(r['split'] for name in ('move','healing') for r in result[name])
    flat = {r['input']: r for r in result['flat']}
    counts = Counter(r['split'] for r in flat.values())
    for seed in range(100000, 120000):
        if all(counts[s] >= budgets[s] for s in budgets):
            break
        state = encounter(seed)
        text = flat_prompt(state)
        split = 'calibration' if int(digest(text)[:8],16) % 4 == 0 else 'teach'
        if text not in flat and counts[split] < budgets[split]:
            decision = rule_decision(state)
            flat[text] = {'input': text, 'split': split, 'label': 'use_item' if decision['action']=='use_item' else decision['chosen_move']}
            revised = rule_decision(state, True)
            flat[text]['corrected_label'] = 'use_item' if revised['action']=='use_item' else revised['chosen_move']
            counts[split] += 1
    if any(counts[s] < budgets[s] for s in budgets):
        raise ValueError('Insufficient unique flat lessons')
    ordered = sorted(flat.values(), key=lambda r: digest(r['input']))
    result['flat'] = []
    for split in ('teach','calibration'):
        result['flat'].extend([r for r in ordered if r['split']==split][:budgets[split]])
    return result


def corrected_rows(rows):
    output = []
    for r in rows:
        values = {t.rsplit('_',1)[0]: bool(int(t.rsplit('_',1)[1])) for t in r['input'].split()[:5]}
        output.append({**r, 'label': healing_label(values, True)})
    return output


class ReviewStop(Exception):
    pass


class Agent:
    def __init__(self, directory=None, mode='network', corrected=False, stop_on_review=False):
        self.mode, self.corrected, self.stop_on_review = mode, corrected, stop_on_review
        self.total_decisions = self.escalations = 0
        self.total_latency_ms = 0
        self.events = []
        if mode == 'network':
            self.move = load(Path(directory) / 'move.s1m')
            self.healing = load(Path(directory) / ('healing-corrected.s1m' if corrected else 'healing.s1m'))
        elif mode == 'single':
            self.flat = load(Path(directory) / ('flat-corrected.s1m' if corrected else 'flat.s1m'))

    def decide(self, state):
        if self.mode == 'rules':
            return rule_decision(state, self.corrected)
        if self.mode == 'single':
            r = self.flat.decide(flat_prompt(state), record_receipt=False)
            selected = r.values['action']
            return {'action': 'use_item' if selected == 'use_item' else 'fight',
                    'chosen_move': selected if selected != 'use_item' else 'move_slot_1',
                    'review': r.is_ambiguous, 'calls': 1, 'sets': r.conformal_sets}
        scores, review = {}, False
        # Legal filtering is shared game physics. It does not rank or fix scores.
        moves = [m for m in state.player_pokemon.moves if m.is_usable()]
        for move in moves:
            r = self.move.decide(move_prompt(state, move), record_receipt=False)
            scores[move.slot] = r.values['preference']
            review |= r.is_ambiguous
        chosen = max(moves, key=lambda m: scores[m.slot]) if moves else None
        h = self.healing.decide(healing_prompt(healing_facts(state, chosen)), record_receipt=False)
        return {'action': h.values['action'], 'chosen_move': chosen.slot if chosen else 'struggle',
                'scores': scores, 'review': bool(review or h.is_ambiguous),
                'calls': len(moves) + 1, 'sets': h.conformal_sets}

    def evaluate(self, state):
        start = time.perf_counter()
        result = self.decide(state)
        allowed, _ = battle_choices(state)
        usable = {m.slot for m in state.player_pokemon.moves if m.is_usable()}
        invalid = result['action'] not in allowed or (result['action'] == 'fight' and usable and result['chosen_move'] not in usable)
        result['review'] |= bool(invalid)
        result['latency_ms'] = (time.perf_counter() - start) * 1000
        result['facts'] = {'hp': state.player_pokemon.current_hp, 'opponent_hp': state.opponent_pokemon.current_hp}
        self.events.append(result)
        self.total_decisions += 1
        self.total_latency_ms += result['latency_ms']
        result['invalid'] = bool(invalid)
        if result['review'] and self.stop_on_review:
            raise ReviewStop()
        # Existing legal-action handling may override invalid predictions. Count
        # them explicitly; never present these as unassisted model decisions.
        telemetry = {**result, 'is_ambiguous': result['review']}
        return telemetry, False, ''  # No scripted advisor or hidden teacher call.


class Battle:
    def __init__(self, state, agent, seed):
        self.state, self.agent, self.seed = state, agent, seed
        self.rng = random.Random(seed).getstate()
        self.initial = asdict(state)
        self.potions_before = sum(state.inventory.values())
        self.stopped = False

    @property
    def done(self):
        return self.stopped or self.state.is_over or self.state.turn_count > 25

    def step(self):
        if self.done:
            return
        previous = random.getstate()
        random.setstate(self.rng)
        try:
            run_battle_simulation(self.state, self.agent, speed='instant', quiet=True, max_turns=self.state.turn_count)
        except ReviewStop:
            self.stopped = True
        finally:
            self.rng = random.getstate()
            random.setstate(previous)

    def result(self):
        events = self.agent.events
        return {'seed': self.seed, 'outcome': 'REVIEW' if self.stopped else self.state.outcome or 'TIMEOUT',
                'turns': len(events), 'potions': self.potions_before - sum(self.state.inventory.values()),
                'review_steps': sum(r['review'] for r in events),
                'invalid_steps': sum(r.get('invalid', False) for r in events),
                'model_calls': sum(r['calls'] for r in events), 'teacher_calls': 0,
                'events': events, 'initial': self.initial, 'final': asdict(self.state)}


def play(directory, seed, family, mode='network', corrected=False, strict=False):
    battle = Battle(encounter(seed, family), Agent(directory, mode, corrected, strict), seed)
    while not battle.done:
        battle.step()
    return battle.result()


def summarize(rows):
    times = [e['latency_ms'] for r in rows for e in r['events']]
    return {'episodes': len(rows), 'wins': sum(r['outcome'] == 'VICTORY' for r in rows),
            'wins_without_review': sum(r['outcome'] == 'VICTORY' and not r['review_steps'] for r in rows),
            'outcomes': dict(Counter(r['outcome'] for r in rows)),
            'turns': sum(r['turns'] for r in rows), 'potions': sum(r['potions'] for r in rows),
            'review_steps': sum(r['review_steps'] for r in rows),
            'invalid_steps': sum(r['invalid_steps'] for r in rows),
            'model_calls': sum(r['model_calls'] for r in rows), 'teacher_calls': 0,
            'median_decision_ms': statistics.median(times) if times else 0}


def prepare(directory):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    data = lessons()
    data['healing-corrected'] = corrected_rows(data['healing'])
    data['flat-corrected'] = [{**r, 'label': r['corrected_label']} for r in data['flat']]
    metadata = {}
    for name, rows in data.items():
        t = time.perf_counter()
        teach(name.split('-')[0], rows, root / f'{name}.s1m')
        metadata[name] = {**dict(Counter(r['split'] for r in rows)), 'bytes': (root / f'{name}.s1m').stat().st_size,
                          'teaching_ms': (time.perf_counter()-t)*1000, 'lessons_sha256': digest(rows)}
    (root / 'lessons.json').write_text(json.dumps(data, indent=2)+'\n')
    metadata['total_ms'] = (time.perf_counter()-started)*1000
    (root / 'teaching.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


def experiment(directory, count=60, seed=8000):
    root = Path(directory)
    metadata = prepare(root)
    results, episodes = {}, {}
    controllers = {'rules': ('rules',False,False), 'single': ('single',False,False),
                   'before': ('network',False,False), 'after': ('network',True,False),
                   'updated_rules': ('rules',True,False), 'updated_single': ('single',True,False), 'after_strict': ('network',True,True)}
    for family in ('healthy','pressure','mixed','survival'):
        results[family], episodes[family] = {}, {}
        for name, (mode, corrected, strict) in controllers.items():
            rows = [play(root, s, family, mode, corrected, strict) for s in range(seed,seed+count)]
            episodes[family][name] = rows
            results[family][name] = summarize(rows)
            print(family, name, results[family][name], flush=True)
    paired = {}
    for family, rows in episodes.items():
        before, after = rows['before'], rows['after']
        paired[family] = {
            'improved_seeds': [a['seed'] for a,b in zip(before,after) if a['outcome']!='VICTORY' and b['outcome']=='VICTORY'],
            'regressed_seeds': [a['seed'] for a,b in zip(before,after) if a['outcome']=='VICTORY' and b['outcome']!='VICTORY'],
            'identical_actions': sum([(e['action'],e['chosen_move']) for e in a['events']]==[(e['action'],e['chosen_move']) for e in b['events']] for a,b in zip(before,after))}
    data = json.loads((root/'lessons.json').read_text())
    changes = dict(Counter(a['split'] for a,b in zip(data['healing'],data['healing-corrected']) if a['label']!=b['label']))
    report = {'experiment': 'Two taught Pokemon battle skills in the existing simplified simulator',
              'seed_start': seed, 'seed_count': count, 'teaching': metadata, 'paired_changes': paired,
              'correction': {'changed_labels': changes, 'unchanged_skill': 'move.s1m'},
              'platform': platform.platform(), 'python': platform.python_version(), 'results': results, 'episodes': episodes,
              'limitations': ['Survival is a constructed stress family for the HP-threshold weakness, not a representative game distribution.',
                             'Equal retained fit/calibration label counts; intermediate supervision and feature representations differ.',
                             'Player always acts first in this simplified simulator; no full-game or Gen-1 fidelity claim.',
                             'Known damage/type rules are shared inputs; the skills do not infer physics from pixels.',
                             'Handwritten original and updated teaching policies are explicit cheaper baselines.',
                             'Flagged predictions execute except in after_strict; legal overrides are counted.',
                             'Familiar local input patterns may recur in unseen encounters; no zero-shot claim.']}
    (root / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report
