"""Bounded real-ROM compatibility check, separate from the battle simulation.

Supports the English Red/Blue binaries identified by pret/pokered/roms.sha1.
RAM layout: pret/pokered/macros/ram.asm (battle_struct) and ram/wram.asm.
Move properties come from the user's ROM; no ROM, save or sprite is distributed.
Observation is privileged RAM access, not visual reasoning. Intro navigation is
scripted. System1 controls the subsequent battle decisions.
"""
from dataclasses import asdict
from hashlib import sha1, sha256
from io import BytesIO
from pathlib import Path
import base64
import time

from ..pokemon_battle_system1 import Pokemon, PokemonMove, BattleState, BattleType
from .battle import Agent

SUPPORTED = {'ea9bcae617fdf159b045185467ae58b2e4a48b9a', 'd7037c83e1ae5b39bde3c30787637ba1d4c48ce2'}
TYPES = {0:'Normal',1:'Fighting',2:'Flying',3:'Poison',4:'Ground',5:'Rock',7:'Bug',8:'Ghost',
         20:'Fire',21:'Water',22:'Grass',23:'Electric',24:'Psychic',25:'Ice',26:'Dragon'}
SPECIES = {177:'Squirtle',153:'Bulbasaur',176:'Charmander'}
MOVE_NAMES = {33:'Tackle',39:'Tail Whip',45:'Growl',10:'Scratch'}
FIGHT = bytes(0x80 + ord(c) - ord('A') for c in 'FIGHT')


def read_state(memory, rom):
    def word(address):
        return memory[address] * 256 + memory[address + 1]
    def pokemon(base):
        hp, maximum = word(base+1), word(base+15)
        if maximum <= 0 or hp > maximum or not 1 <= memory[base+14] <= 100:
            raise ValueError('Battle RAM is not initialized')
        types = tuple(dict.fromkeys(TYPES[memory[base+i]] for i in (5,6)))
        moves = []
        for i in range(4):
            move_id = memory[base+8+i]
            if not move_id:
                continue
            if not 1 <= move_id <= 165:
                raise ValueError('Unknown move id')
            start = 0x38000 + (move_id-1) * 6
            _, effect, power, kind, accuracy, pp = rom[start:start+6]
            category = 'status' if not power else 'special' if kind >= 20 else 'physical'
            moves.append(PokemonMove(f'move_slot_{i+1}', MOVE_NAMES.get(move_id, f'Move {move_id}'),
                                    TYPES[kind], power, round(accuracy*100/255), memory[base+25+i] & 63, pp, category))
        status = memory[base+4]
        return Pokemon(SPECIES.get(memory[base],f'Species {memory[base]}'), memory[base+14], hp, maximum,
                       types, moves, attack=word(base+17), defense=word(base+19),
                       speed=word(base+21), special=word(base+23),
                       status='OK' if status==0 else f'STATUS_{status}')
    mode = memory[0xd057]
    if mode not in (1,2):
        raise ValueError('Not in a battle')
    if memory[0xd31d] > 20:
        raise ValueError('Invalid bag contents')
    inventory = {}
    for i in range(memory[0xd31d]):
        item, count = memory[0xd31e+2*i], memory[0xd31f+2*i]
        if item in (0x14,0x13):
            inventory['Potion' if item==0x14 else 'Super Potion'] = count
    return BattleState(pokemon(0xd014), pokemon(0xcfe5), BattleType.TRAINER if mode==2 else BattleType.WILD,
                       party=[], inventory=inventory, can_run=mode==1)


class RomBattle:
    def __init__(self, rom_path, directory, corrected=True, *, checkpoint=None,
                 mode='network', strict=False, fixture=None):
        from pyboy import PyBoy
        self.rom = Path(rom_path).read_bytes()
        if sha1(self.rom).hexdigest() not in SUPPORTED:
            raise ValueError('Use an unmodified English Pokemon Red or Blue ROM')
        # File-like inputs prevent reading or overwriting the user's battery save.
        self.emulator = PyBoy(BytesIO(self.rom), window='null', sound_emulated=False)
        self.emulator.set_emulation_speed(0)
        self.agent = Agent(directory, mode=mode, corrected=corrected, stop_on_review=strict)
        files = (['move.s1m', 'healing-corrected.s1m' if corrected else 'healing.s1m'] if mode == 'network'
                 else ['flat-corrected.s1m' if corrected else 'flat.s1m'] if mode == 'single' else [])
        self.skill_hashes = {name: sha256((Path(directory)/name).read_bytes()).hexdigest() for name in files}
        self.fixture = fixture
        self.events = []
        self.done = False
        self.outcome = 'READY'
        self.started = time.perf_counter()
        try:
            if checkpoint is None:
                from ..pokemon_gameboy_gui import fast_forward_to_rival_battle
                fast_forward_to_rival_battle(self.emulator)
                for button in ('a','b','start','select','up','down','left','right'):
                    self.emulator.button_release(button)
                self.advance_to_menu()
            else:
                self.emulator.load_state(BytesIO(checkpoint))
                if not self.at_menu():
                    raise ValueError('Checkpoint must be at a battle menu')
            if self.done:
                raise ValueError('Could not reach the rival battle menu')
            self.restart()
        except Exception:
            self.close()
            raise

    def at_menu(self):
        return FIGHT in bytes(self.emulator.memory[0xc3a0:0xc508])

    def save_checkpoint(self):
        stream = BytesIO()
        self.emulator.save_state(stream)
        return stream.getvalue()

    def restart(self, checkpoint=None):
        if checkpoint is not None:
            self.emulator.load_state(BytesIO(checkpoint))
        if not self.at_menu():
            raise ValueError('Checkpoint must be at a battle menu')
        self.initial = asdict(read_state(self.emulator.memory, self.rom))
        self.checkpoint_sha256 = sha256(self.save_checkpoint()).hexdigest()
        self.events = []
        self.done, self.outcome = False, 'READY'
        self.started = time.perf_counter()

    def press(self, button, frames=8):
        self.emulator.button(button,4)
        self.emulator.tick(frames)

    def advance_to_menu(self):
        e = self.emulator
        for frame in range(5000):
            e.tick()
            if self.at_menu():
                e.button_release('a')
                e.tick(5)
                return
            if self.events:
                # Read the actual HP bytes, never infer victory from a screenshot.
                if e.memory[0xcfe6] == e.memory[0xcfe7] == 0:
                    self.done, self.outcome = True, 'VICTORY'
                    # HP is written before the bar/faint animation. Let the
                    # display catch up without advancing text with a button.
                    e.tick(240)
                    return
                if e.memory[0xd015] == e.memory[0xd016] == 0:
                    self.done, self.outcome = True, 'DEFEAT'
                    e.tick(240)
                    return
            if frame % 90 == 0:
                self.press('a')
        raise RuntimeError('Timed out waiting for the battle menu')

    def use_item(self, state):
        # Match the explicit item preference used by healing_facts(). This is
        # button dispatch, not a second policy deciding whether to heal.
        item_id, name = (0x13, 'Super Potion') if state.inventory.get('Super Potion', 0) else (0x14, 'Potion')
        memory = self.emulator.memory
        if not state.inventory.get(name, 0) or state.player_pokemon.current_hp >= state.player_pokemon.max_hp:
            raise ValueError('Selected healing item is unavailable or cannot help')
        index = next(i for i in range(memory[0xd31d]) if memory[0xd31e+2*i] == item_id)
        self.press('left'); self.press('down'); self.press('a', 30)
        # Bag cursors are zero-based and include a scrolling offset.
        for _ in range(25):
            current = memory[0xcc26] + memory[0xcc36]
            if current == index:
                break
            self.press('down' if current < index else 'up')
        if memory[0xcc26] + memory[0xcc36] != index or self.at_menu():
            raise RuntimeError('Bag menu did not reach the requested item')
        self.press('a', 30)
        # Use the item on the observed active party member, never a fixed slot.
        active = memory[0xcc2f]
        for _ in range(6):
            current = memory[0xcc26]
            if current == active:
                break
            self.press('down' if current < active else 'up')
        if memory[0xcc26] != active:
            raise RuntimeError('Party menu did not reach the active Pokemon')
        self.press('a', 30)
        return name

    def step(self, *, screen=True):
        if self.done:
            return self.snapshot(screen=screen)
        if len(self.events) >= 30:
            self.done, self.outcome = True, 'TIMEOUT'
            return self.snapshot(screen=screen)
        state = read_state(self.emulator.memory, self.rom)
        start = time.perf_counter()
        decision = self.agent.decide(state)
        decision['latency_ms'] = (time.perf_counter()-start)*1000
        decision['before'] = asdict(state)
        decision['executed'] = False
        self.events.append(decision)
        if decision['review'] and self.agent.stop_on_review:
            self.done, self.outcome = True, 'REVIEW'
            return self.snapshot(screen=screen)
        if decision['action'] not in ('fight', 'use_item'):
            self.done, self.outcome = True, 'UNSUPPORTED_ACTION'
            return self.snapshot(screen=screen)
        if decision['action'] == 'fight' and decision['chosen_move'] not in {m.slot for m in state.player_pokemon.moves if m.is_usable()}:
            self.done, self.outcome = True, 'UNSUPPORTED_MOVE'
            return self.snapshot(screen=screen)
        if decision['action'] == 'use_item':
            if not any(state.inventory.values()) or state.player_pokemon.current_hp >= state.player_pokemon.max_hp:
                self.done, self.outcome = True, 'UNSUPPORTED_ITEM'
                return self.snapshot(screen=screen)
            decision['item'] = self.use_item(state)
        else:
            # Navigate the actual battle menu and then the observed move cursor.
            self.press('up'); self.press('left'); self.press('a',20)
            target = int(decision['chosen_move'][-1])
            for _ in range(4):
                current = self.emulator.memory[0xcc26]
                if current == target:
                    break
                self.press('down' if current < target else 'up')
            if self.emulator.memory[0xcc26] != target:
                raise RuntimeError('Move menu did not reach the requested slot')
            self.press('a',20)
        decision['executed'] = True
        self.advance_to_menu()
        decision['after_hp'] = self.emulator.memory[0xd015]*256+self.emulator.memory[0xd016]
        decision['after_enemy_hp'] = self.emulator.memory[0xcfe6]*256+self.emulator.memory[0xcfe7]
        decision['after_inventory'] = read_state(self.emulator.memory, self.rom).inventory
        if decision['action'] == 'use_item':
            name = decision['item']
            if state.inventory[name] - decision['after_inventory'].get(name, 0) != 1:
                raise RuntimeError('The game did not consume exactly one requested potion')
        return self.snapshot(screen=screen)

    def png(self):
        stream = BytesIO()
        self.emulator.screen.image.save(stream,format='PNG')
        return stream.getvalue()

    def snapshot(self, *, screen=True):
        result = {'done':self.done,'outcome':self.outcome,'turns':sum(e['executed'] for e in self.events),
                'potions':sum(e.get('item') is not None and e['executed'] for e in self.events),
                'review_steps':sum(e['review'] for e in self.events),
                'decision':self.events[-1] if self.events else None,
                'fixture':self.fixture, 'corrected':self.agent.corrected, 'strict':self.agent.stop_on_review,
                'checkpoint_sha256':self.checkpoint_sha256}
        if screen:
            result['screen'] = 'data:image/png;base64,'+base64.b64encode(self.png()).decode()
        return result

    def report(self):
        return {'experiment':'Actual English Pokemon Red/Blue battle engine',
                'rom_sha256':sha256(self.rom).hexdigest(), 'skill_sha256':self.skill_hashes, 'initial':self.initial,
                **self.snapshot(screen=False), 'events':self.events,
                'mode':self.agent.mode, 'model_calls':sum(e['calls'] for e in self.events),
                'review_steps':sum(e['review'] for e in self.events), 'teacher_calls':0,
                'elapsed_seconds':time.perf_counter()-self.started,
                'limitations':['Scripted intro navigation; taught controller starts at battle menu.',
                               'Reads RAM and move properties from ROM; not vision or full-game play.',
                               'Controlled fixtures edit the initial RAM only; subsequent turns use game physics.' if self.fixture else
                               'Unmodified starter battle has no healing items.',
                               'Stops before flagged actions.' if self.agent.stop_on_review else 'Flagged predictions execute and are counted.',
                               'Same initial RNG state; different button/action sequences can cause different later random draws.']}

    def close(self):
        self.emulator.stop(save=False)
