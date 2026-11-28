#!/usr/bin/env python3
"""Writes an Educore Stories lesson from a topic, using Claude.

    python3 scripts/make_story.py "photosynthesis" --age 10

It asks Claude for a story in the format godot/README.md describes, checks it
the way the editor's problem list does, plays it through to make sure a
student can reach an ending, and asks Claude to fix whatever the checks find.
Then it saves the story file and prints a link students can open.

Needs `pip install anthropic` and Claude API credentials (ANTHROPIC_API_KEY).
`--check FILE` checks an existing story without calling Claude.
"""

import argparse
import base64
import json
import math
import os
import re
import sys
import zlib
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SAMPLE = REPO / "godot" / "stories" / "water_cycle.json"

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_SITE = "https://matoo125.github.io/education-tool/stories/"
# Models whose declined requests the API can retry on a fallback model.
FALLBACK_MODELS = {"claude-fable-5-2", "claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}

# These mirror godot/scripts/story.gd.
ANIMALS = ["dog", "cat", "chicken", "duck", "sheep", "cow", "rabbit", "fox", "owl"]
PROP_KINDS = [
    "house", "tree", "pine", "rock", "fence", "lamp",
    "bench", "well", "crate", "table", "shelf", "plant",
    "pond", "river", "bridge", "fountain",
] + ANIMALS
WEATHERS = ["clear", "cloudy", "rain"]
CHOICE_FIELDS = ["requires", "unless", "gives", "takes", "sets", "ending"]
DOOR_SPAWN_DISTANCE = 2.6
# Props a student can't walk through, and how much room each takes (metres).
BLOCKERS = {"house": 2.9, "well": 1.4, "pond": 2.8, "river": 2.2, "fountain": 1.8}

# How many game states the play-through may visit before giving up.
MAX_STATES = 200_000


# The story format ----------------------------------------------------------

def _text(d, key, fallback=""):
    value = d.get(key)
    d[key] = fallback if value is None else str(value)


def _list(value):
    return [entry for entry in value if isinstance(entry, dict)] if isinstance(value, list) else []


def _pos(value):
    if isinstance(value, list) and len(value) >= 2:
        try:
            return [float(value[0]), float(value[1])]
        except (TypeError, ValueError):
            pass
    return [0.0, 0.0]


def _number(value, fallback):
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def normalized(story):
    """The story with every field filled in, as story.gd's normalized() does."""
    s = json.loads(json.dumps(story))
    _text(s, "title", "Untitled story")
    _text(s, "intro")
    _text(s, "goal")
    for key in ("places", "characters", "items"):
        s[key] = _list(s.get(key))
    for place in s["places"]:
        _text(place, "id")
        _text(place, "name", "Unnamed place")
        place["indoor"] = bool(place.get("indoor", False))
        size = _number(place.get("size"), 16.0 if place["indoor"] else 36.0)
        place["size"] = min(max(size, 8.0), 120.0)
        _text(place, "weather", "clear")
        if place["weather"] not in WEATHERS:
            place["weather"] = "clear"
        place["props"] = _list(place.get("props"))
        for prop in place["props"]:
            _text(prop, "kind", "tree")
            if prop["kind"] not in PROP_KINDS:
                prop["kind"] = "crate"
            prop["pos"] = _pos(prop.get("pos"))
            prop["rot"] = _number(prop.get("rot"), 0.0)
        place["doors"] = _list(place.get("doors"))
        for door in place["doors"]:
            door["pos"] = _pos(door.get("pos"))
            door["rot"] = _number(door.get("rot"), 0.0)
            _text(door, "to")
            _text(door, "label")
    for character in s["characters"]:
        _text(character, "id")
        _text(character, "name", "Someone")
        _text(character, "place")
        character["pos"] = _pos(character.get("pos"))
        _text(character, "look", "person")
        if character["look"] not in ANIMALS:
            character["look"] = "person"
        character["lines"] = _list(character.get("lines"))
        for i, line in enumerate(character["lines"]):
            _text(line, "text")
            line["opens"] = bool(line.get("opens", i == 0))
            _text(line, "requires")
            _text(line, "unless")
            line["choices"] = _list(line.get("choices"))
            for choice in line["choices"]:
                _text(choice, "text", "...")
                choice["next"] = int(_number(choice.get("next"), -1))
                for key in CHOICE_FIELDS:
                    _text(choice, key)
    for item in s["items"]:
        _text(item, "id")
        _text(item, "name", "Thing")
        _text(item, "description")
        _text(item, "place")
        item["pos"] = _pos(item.get("pos"))
    _text(s, "start_place")
    if s["places"] and s["start_place"] not in [p["id"] for p in s["places"]]:
        s["start_place"] = s["places"][0]["id"]
    s["start_pos"] = _pos(s.get("start_pos"))
    return s


def tidy(value):
    """Drops empty text fields, so the file stays readable and Godot fills in
    its own defaults (an empty colour would not be one)."""
    if isinstance(value, dict):
        return {k: tidy(v) for k, v in value.items() if v != "" and v is not None}
    if isinstance(value, list):
        return [tidy(v) for v in value]
    return value


def find(story, key, id):
    return next((entry for entry in story[key] if entry["id"] == id), None)


def forward(rot):
    return math.sin(math.radians(rot)), math.cos(math.radians(rot))


def arrival(door):
    """Where a student appears after coming in through `door`."""
    fx, fz = forward(door["rot"])
    return [door["pos"][0] + fx * DOOR_SPAWN_DISTANCE, door["pos"][1] + fz * DOOR_SPAWN_DISTANCE]


# Checks --------------------------------------------------------------------

def problems(story):
    """The editor's problem list (story.gd's problems()), plus duplicate IDs."""
    out = []
    if not story["places"]:
        return ["The story has no places."]
    for key in ("places", "characters", "items"):
        ids = [entry["id"] for entry in story[key]]
        for id in sorted(set(ids)):
            if id == "":
                out.append(f"Something in {key} has no id.")
            elif ids.count(id) > 1:
                out.append(f'Two {key} share the id "{id}".')
    place_ids = {place["id"] for place in story["places"]}
    item_ids = {item["id"] for item in story["items"]}
    for item in story["items"]:
        if item["place"] != "" and item["place"] not in place_ids:
            out.append(f'{item["name"]} is in a place that doesn\'t exist.')
    flags = {
        choice["sets"]
        for character in story["characters"]
        for line in character["lines"]
        for choice in line["choices"]
        if choice["sets"] != ""
    }
    for clash in sorted(flags & item_ids):
        out.append(f'"{clash}" is both an item and a fact; give them different names.')
    known = item_ids | flags
    for place in story["places"]:
        for door in place["doors"]:
            if door["to"] not in place_ids:
                out.append(f'A door in {place["name"]} doesn\'t lead anywhere.')
            elif door["to"] == place["id"]:
                out.append(f'A door in {place["name"]} leads back into itself.')
    has_ending = False
    for character in story["characters"]:
        who = character["name"]
        if character["place"] not in place_ids:
            out.append(f"{who} isn't in any place.")
        lines = character["lines"]
        if not lines:
            out.append(f"{who} has nothing to say.")
            continue
        opens = False
        for i, line in enumerate(lines):
            where = f"{who}, line {i + 1}"
            if line["opens"]:
                opens = True
            for key in ("requires", "unless"):
                if line["opens"] and line[key] != "" and line[key] not in known:
                    out.append(f'{where}: nothing gives or sets "{line[key]}".')
            for choice in line["choices"]:
                if choice["ending"] != "":
                    has_ending = True
                if choice["next"] >= len(lines):
                    out.append(f'{where}: a choice goes to line index {choice["next"]}, which doesn\'t exist.')
                for key in ("requires", "unless"):
                    if choice[key] != "" and choice[key] not in known:
                        out.append(f'{where}: nothing gives or sets "{choice[key]}".')
                for key in ("gives", "takes"):
                    if choice[key] != "" and choice[key] not in item_ids:
                        out.append(f'{where}: "{choice[key]}" isn\'t an item.')
        if not opens:
            out.append(f"{who} has no line that can start a conversation.")
    if not has_ending:
        out.append("No choice ends the story.")
    return out


def layout_problems(story):
    """Things that would be out of reach or stuck inside something."""
    out = []
    for place in story["places"]:
        half = place["size"] / 2.0
        name = place["name"]
        blockers = [(p["pos"], BLOCKERS[p["kind"]], p["kind"]) for p in place["props"] if p["kind"] in BLOCKERS]

        def check(what, pos, margin, blocked=True):
            if abs(pos[0]) > half - margin or abs(pos[1]) > half - margin:
                out.append(
                    f"{what} in {name} is at {_fmt(pos)}, outside the place"
                    f" (keep x and z within ±{half - margin:g})."
                )
                return
            if not blocked:
                return
            for center, radius, kind in blockers:
                if math.dist(pos, center) < radius:
                    out.append(f"{what} in {name} is at {_fmt(pos)}, inside the {kind} at {_fmt(center)}.")
                    return

        for prop in place["props"]:
            check(f'A {prop["kind"]}', prop["pos"], 0.0, blocked=False)
        for door in place["doors"]:
            check(f'The door to "{door["to"]}"', door["pos"], 0.0, blocked=False)
            check(
                f'The spot in front of the door to "{door["to"]}" (where students arrive, {DOOR_SPAWN_DISTANCE} m'
                f' along its rot {door["rot"]:g})',
                arrival(door), 0.5,
            )
        for character in story["characters"]:
            if character["place"] == place["id"]:
                check(character["name"], character["pos"], 1.0)
        for item in story["items"]:
            if item["place"] == place["id"]:
                check(item["name"], item["pos"], 1.0)
        if story["start_place"] == place["id"]:
            check("The start point", story["start_pos"], 1.0)
    return out


def _fmt(pos):
    return f"[{pos[0]:g}, {pos[1]:g}]"


def _allows(entry, inventory, flags):
    def has(fact):
        return fact in inventory or fact in flags

    return (entry["requires"] == "" or has(entry["requires"])) and (entry["unless"] == "" or not has(entry["unless"]))


def walkthrough(story):
    """The shortest way to an ending, as a list of steps a student takes.

    Returns None when no ending can be reached. Follows story_state.gd: where
    students are, what they carry, which facts are set, what they've picked up.
    """
    doors = {place["id"]: [d["to"] for d in place["doors"]] for place in story["places"]}
    place_names = {place["id"]: place["name"] for place in story["places"]}
    item_names = {item["id"]: item["name"] for item in story["items"]}
    start = (story["start_place"], frozenset(), frozenset(), frozenset())
    came_from = {start: None}
    queue = deque([start])

    def steps_to(state, last):
        steps = [last]
        while came_from[state] is not None:
            state, step = came_from[state]
            steps.append(step)
        return steps[::-1]

    while queue and len(came_from) < MAX_STATES:
        state = queue.popleft()
        place, inventory, flags, collected = state
        moves = []
        for to in doors.get(place, []):
            if to in doors:
                moves.append(((to, inventory, flags, collected), f"Go to {place_names[to]}."))
        for item in story["items"]:
            if item["place"] == place and item["id"] not in collected:
                moves.append((
                    (place, inventory | {item["id"]}, flags, collected | {item["id"]}),
                    f'Pick up {item["name"]}.',
                ))
        for character in story["characters"]:
            if character["place"] != place:
                continue
            for outcome, said, ending in _conversations(character, inventory, flags, collected):
                step = f'Talk to {character["name"]}: ' + " / ".join(f'"{s}"' for s in said)
                if ending:
                    return steps_to(state, step)
                moves.append(((place,) + outcome, step))
        for next_state, step in moves:
            if next_state not in came_from:
                came_from[next_state] = (state, step)
                queue.append(next_state)
    return None


def _conversations(character, inventory, flags, collected):
    """Every way one talk with `character` can go: (state after, choices said, ended the story)."""
    lines = character["lines"]
    opening = next((i for i, line in enumerate(lines) if line["opens"] and _allows(line, inventory, flags)), -1)
    if opening < 0:
        return []
    results = {}
    seen = set()
    stack = [(opening, inventory, flags, collected, ())]
    while stack:
        index, inv, fl, col, said = stack.pop()
        if (index, inv, fl, col) in seen:
            continue
        seen.add((index, inv, fl, col))
        for choice in lines[index]["choices"]:
            if not _allows(choice, inv, fl):
                continue
            inv2, fl2, col2 = inv, fl, col
            if choice["gives"] != "":
                inv2, col2 = inv2 | {choice["gives"]}, col2 | {choice["gives"]}
            if choice["takes"] != "":
                inv2 = inv2 - {choice["takes"]}
            if choice["sets"] != "":
                fl2 = fl2 | {choice["sets"]}
            said2 = said + (choice["text"],)
            if choice["ending"] != "":
                return [((inv2, fl2, col2), said2, True)]
            if 0 <= choice["next"] < len(lines):
                stack.append((choice["next"], inv2, fl2, col2, said2))
            else:
                results.setdefault((inv2, fl2, col2), said2)
        # A line with no choices a student can pick just ends the talk.
    return [(outcome, said, False) for outcome, said in results.items()]


def check(story):
    """Every problem found in a normalized story, and a walkthrough if there is one."""
    found = problems(story) + layout_problems(story)
    steps = None
    if not found:
        steps = walkthrough(story)
        if steps is None:
            found.append(
                "Playing the story through, no ending can be reached: follow the doors, items and"
                " conditions from the start and make sure a student can get to a choice with an ending."
            )
    return found, steps


# Story links ---------------------------------------------------------------

def to_code(story):
    """The story packed the way story.gd's to_code() packs it (deflate, URL-safe base64)."""
    packed = zlib.compress(json.dumps(story, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), 9)
    return base64.urlsafe_b64encode(packed).decode("ascii").rstrip("=")


def from_code(code):
    packed = base64.urlsafe_b64decode(code + "=" * (-len(code) % 4))
    return json.loads(zlib.decompress(packed).decode("utf-8"))


def link_for(story, site=DEFAULT_SITE):
    return site.rstrip("/") + "/#story=" + to_code(story)


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "story"


# Asking Claude -------------------------------------------------------------

SYSTEM_PROMPT = """\
You write lessons for Educore Stories, a tool teachers use to turn a lesson into a small 3D story game. \
A teacher gives you a topic and the age of their students. You write one complete story, as JSON, that the \
teacher can hand to a class as a link. The teacher may not read it closely first, so it has to be factually \
right, playable from start to finish, and written at the students' reading level.

## How a story plays

A student walks around small 3D places, talks to characters, picks up items and goes through doors. \
The story ends when the student picks a dialogue choice that has an "ending" text.

- Talking: a conversation starts at the first line of the character, in order, that has "opens": true and \
whose conditions hold. The student then picks a choice. A choice's "next" is the index (from 0) of the \
character's line to show next, or -1 to stop talking. A choice can also give an item ("gives"), take one \
away ("takes"), set a fact ("sets"), or end the story ("ending").
- Conditions: "requires" and "unless" on a line or a choice each name one item id or one fact. "requires" \
holds when the student carries that item or the fact is set; "unless" holds when they don't. Facts are just \
names that "sets" creates; they never get unset. Use them so people say different things as the story moves on.
- Items lie in a place until picked up (or have an empty "place" and are given in conversation). When a \
student picks one up they read its "description", so that is the natural place for the lesson content itself.
- Doors lead to another place. Students arrive in the new place in front of the door that leads back, so \
every place needs a door back to where students came from.

## What makes a good lesson story

- Give the student a reason to care: someone has a problem that the lesson's knowledge solves.
- Teach before testing. The student should meet the facts (an item's description, a character explaining) \
before anyone asks about them.
- Check understanding in conversation: a character asks a question, one choice is right and the others are \
plausible mistakes a student of this age would really make, not jokes. A wrong answer leads to a short, kind \
hint and back to the question; it never ends the story or blocks progress.
- Gate progress with facts and items so the order makes sense: the final conversation should need a fact that \
is only set after the student answered correctly.
- The ending text says what happened in the story and restates, in a sentence or two, what the student learned.
- Aim for 2 or 3 places, 3 to 5 characters, 1 to 3 items, and 2 to 4 questions that check understanding. \
About ten minutes of play.
- Keep each spoken line to at most three short sentences, and each choice to one sentence. Write everything \
students read in the language the teacher asks for; keep ids in lowercase English with underscores.
- "intro" is shown first: two or three sentences that set the scene. Don't explain the controls; the game \
does. "goal" is one sentence that stays on screen.
- Everything must be true. If you simplify for the age group, simplify without saying anything false.
- Every story is vegan friendly: nobody in it uses animals. No one eats, buys, sells or produces meat, fish, \
milk, eggs, cheese, honey, wool or leather, and no one farms, hunts, fishes, rides or works animals. Food is \
plant-based (fruit, vegetables, bread, nuts, oat milk). Animals are free: wildlife, companions, residents of \
a sanctuary, or characters who talk. This also covers word lists and examples, and the animal props, which \
must not be dressed as livestock (a fenced herd, a cow beside a milk stall).

## Building places

Positions are [x, z] in metres on the ground, with [0, 0] the middle of the place; +x is to the student's \
right and +z is towards the camera. "size" is the length of the place's sides, so things must stay within \
plus or minus half of it.

- Outdoor places: size 30 to 40. Indoor places ("indoor": true): size 14 to 18; they get walls automatically.
- Prop kinds: house, tree, pine, rock, fence, lamp, bench, well, crate, table, shelf, plant, pond, river, \
bridge, fountain, and animals that wander: dog, cat, chicken, duck, sheep, cow, rabbit, fox, owl. There are \
no other kinds, so set the scene with these: a "lab" is an indoor place with tables, shelves and plants.
- Dress each place with 12 to 30 props so it doesn't look empty: trees and rocks around the edges of outdoor \
places, furniture along the walls indoors. Leave the middle open for walking.
- Students can't walk through houses, wells, ponds, fountains or rivers. Keep characters, items, the start \
point and the spots in front of doors at least 3.5 m from a house or pond, and 2 m from the others. Rivers \
block the way entirely unless a bridge crosses them, so avoid rivers unless the lesson is about one.
- Keep characters and items at least 2 m inside the place's edge and 2 m apart from each other.
- Doors: "rot" is the direction the door faces, in degrees: 0 faces +z, 90 faces +x, 180 faces -z, -90 \
faces -x. Students arrive 2.6 m in front of the door, so the door must face open floor. Outdoors, a door \
into a building stands just in front of a house prop (for a house at [11, -8], a door at [11, -5.4] with \
rot 0). Indoors, the door back sits on a wall facing in (for size 16, a door at [0, 7.6] with rot 180). \
"label" is the name of the place it leads to.
- A character's "look" is "person" or an animal kind, for a talking animal. Give characters different colours.
- "weather" is clear, cloudy or rain (outdoor places only). Colours are "#rrggbb"; leave a prop's colour \
empty for its usual colour, and "ground" and "sky" empty for the usual grass and sky.
- Use an empty string for any text field you don't need (conditions, effects, an item's place when a \
character gives it).

## An example

This is a complete story for 9-year-olds about the water cycle. Match its scale and how it is built, but \
give your story its own setting, characters and plot.

"""

_STR = {"type": "string"}
_POS = {"type": "array", "items": {"type": "number"}, "description": "[x, z] in metres"}


def _object(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


STORY_SCHEMA = _object({
    "title": _STR,
    "intro": _STR,
    "goal": _STR,
    "start_place": _STR,
    "start_pos": _POS,
    "places": {"type": "array", "items": _object({
        "id": _STR,
        "name": _STR,
        "indoor": {"type": "boolean"},
        "size": {"type": "number"},
        "ground": _STR,
        "sky": _STR,
        "weather": {"type": "string", "enum": WEATHERS},
        "props": {"type": "array", "items": _object({
            "kind": {"type": "string", "enum": PROP_KINDS},
            "pos": _POS,
            "rot": {"type": "number"},
            "color": _STR,
        })},
        "doors": {"type": "array", "items": _object({
            "pos": _POS,
            "rot": {"type": "number"},
            "to": _STR,
            "label": _STR,
        })},
    })},
    "characters": {"type": "array", "items": _object({
        "id": _STR,
        "name": _STR,
        "place": _STR,
        "pos": _POS,
        "rot": {"type": "number"},
        "color": _STR,
        "look": {"type": "string", "enum": ["person"] + ANIMALS},
        "lines": {"type": "array", "items": _object({
            "text": _STR,
            "opens": {"type": "boolean"},
            "requires": _STR,
            "unless": _STR,
            "choices": {"type": "array", "items": _object({
                "text": _STR,
                "next": {"type": "integer"},
                "requires": _STR,
                "unless": _STR,
                "gives": _STR,
                "takes": _STR,
                "sets": _STR,
                "ending": _STR,
            })},
        })},
    })},
    "items": {"type": "array", "items": _object({
        "id": _STR,
        "name": _STR,
        "place": _STR,
        "pos": _POS,
        "color": _STR,
        "description": _STR,
    })},
})


class StoryError(Exception):
    pass


def system_prompt():
    return SYSTEM_PROMPT + SAMPLE.read_text(encoding="utf-8")


def request_text(topic, age, language, notes=""):
    text = f"Topic: {topic}\nStudents' age: {age}\nLanguage: {language}\n"
    if notes:
        text += f"Teacher's notes: {notes}\n"
    return text + "\nWrite the story."


def generate(topic, age=10, language="English", notes="", model=DEFAULT_MODEL, effort="high",
             attempts=3, client=None, log=print):
    """Asks Claude for a story until it passes check(), at most `attempts` times.

    Returns (story, problems, walkthrough): the last story it wrote, normalized
    and tidied, the problems left in it (none when it passed), and the steps to
    an ending.
    """
    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    messages = [{"role": "user", "content": request_text(topic, age, language, notes)}]
    story, found, steps = None, [], None
    for attempt in range(1, attempts + 1):
        log(f"Asking {model} for the story (attempt {attempt} of {attempts})...")
        response = _ask(client, model, effort, messages)
        usage = response.usage
        log(f"  {usage.input_tokens} input and {usage.output_tokens} output tokens"
            f" ({getattr(usage, 'cache_read_input_tokens', 0) or 0} read from cache)")
        if response.stop_reason == "refusal":
            raise StoryError("Claude declined to write this story.")
        if response.stop_reason == "max_tokens":
            raise StoryError("The story was cut off before it was finished; try a narrower topic.")
        text = next((block.text for block in response.content if block.type == "text"), "")
        try:
            story = normalized(json.loads(text))
        except (json.JSONDecodeError, AttributeError) as error:
            raise StoryError(f"Claude's reply wasn't a story ({error}).")
        found, steps = check(story)
        story = tidy(story)
        if not found:
            break
        log(f"  The checks found {len(found)} problem(s):")
        for problem in found:
            log(f"    - {problem}")
        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": "I ran the story through the game's checks and they found these problems:\n"
            + "\n".join(f"- {problem}" for problem in found)
            + "\n\nFix them and write the whole story again.",
        })
    return story, found, steps


def _ask(client, model, effort, messages):
    request = {
        "model": model,
        "max_tokens": 64000,
        "system": [{"type": "text", "text": system_prompt(), "cache_control": {"type": "ephemeral"}}],
        "messages": messages,
        "output_config": {"format": {"type": "json_schema", "schema": STORY_SCHEMA}},
    }
    if effort and not model.startswith("claude-haiku-4"):
        request["output_config"]["effort"] = effort
    if model in FALLBACK_MODELS:
        # On a safety decline, the API retries on its recommended fallback
        # model instead of returning the refusal.
        request["betas"] = ["server-side-fallback-2026-07-01"]
        request["fallbacks"] = "default"
        stream = client.beta.messages.stream(**request)
    else:
        stream = client.messages.stream(**request)
    # Streamed, because a whole story can take longer than a plain request may.
    with stream as events:
        return events.get_final_message()


# Command line --------------------------------------------------------------

def report(story, found, steps, log=print):
    if found:
        log(f"{len(found)} problem(s):")
        for problem in found:
            log(f"  - {problem}")
    elif steps:
        log(f"A student can finish it in {len(steps)} steps:")
        for i, step in enumerate(steps, 1):
            log(f"  {i}. {step}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Write an Educore Stories lesson from a topic, using Claude.")
    parser.add_argument("topic", nargs="?", help='what the lesson is about, e.g. "photosynthesis"')
    parser.add_argument("--age", type=int, default=10, help="the students' age (default 10)")
    parser.add_argument("--language", default="English", help="the language students read (default English)")
    parser.add_argument("--notes", default="", help="anything else the story should cover or avoid")
    parser.add_argument("--out", help="where to save the story (default godot/stories/<title>.json)")
    parser.add_argument("--site", default=os.environ.get("EDUCORE_STORIES_URL", DEFAULT_SITE),
                        help="the web address of Educore Stories, for the student link")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--attempts", type=int, default=3, help="how many times Claude may try (default 3)")
    parser.add_argument("--check", metavar="FILE", help="check an existing story file instead of writing one")
    args = parser.parse_args(argv)

    if args.check:
        story = normalized(json.loads(Path(args.check).read_text(encoding="utf-8")))
        found, steps = check(story)
        print(f'"{story["title"]}"')
        report(story, found, steps)
        if not found:
            print("\nStudent link:\n" + link_for(tidy(story), args.site))
        return 1 if found else 0
    if not args.topic:
        parser.error("give a topic, or --check FILE")

    try:
        story, found, steps = generate(args.topic, args.age, args.language, args.notes,
                                       args.model, args.effort, args.attempts)
    except ImportError:
        print("This needs the Anthropic SDK: pip install anthropic", file=sys.stderr)
        return 2
    except StoryError as error:
        print(error, file=sys.stderr)
        return 2
    except Exception as error:
        # The SDK's own errors: a missing or wrong key, no network, rate limits.
        if type(error).__module__.split(".")[0] != "anthropic":
            raise
        print(f"The request to Claude failed: {error}", file=sys.stderr)
        return 2

    out = Path(args.out) if args.out else REPO / "godot" / "stories" / (slug(story["title"]) + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(story, indent="\t", ensure_ascii=False) + "\n", encoding="utf-8")
    print(f'\nSaved "{story["title"]}" to {out}')
    report(story, found, steps)
    if found:
        print("\nThe story still has problems; open it in the editor to fix them, or run this again.")
        return 1
    print("\nStudent link:\n" + link_for(story, args.site))
    return 0


if __name__ == "__main__":
    sys.exit(main())
