"""Tests for make_story.py that run without calling Claude."""

import copy
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_story  # noqa: E402


def sample():
    return make_story.normalized(json.loads(make_story.SAMPLE.read_text(encoding="utf-8")))


class FakeClient:
    """Stands in for anthropic.Anthropic: replies with the given stories in turn."""

    def __init__(self, stories, stop_reason="end_turn"):
        self.replies = [json.dumps(story) for story in stories]
        self.stop_reason = stop_reason
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))
        self.messages = SimpleNamespace(stream=self._stream)

    def _stream(self, **request):
        self.requests.append(copy.deepcopy(request))
        message = SimpleNamespace(
            stop_reason=self.stop_reason,
            content=[SimpleNamespace(type="thinking", thinking=""),
                     SimpleNamespace(type="text", text=self.replies.pop(0))],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1, cache_read_input_tokens=0),
        )
        client = self

        class Stream:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get_final_message(self):
                return message

        client.last = message
        return Stream()


class Checks(unittest.TestCase):
    def test_sample_story_passes_and_can_be_finished(self):
        found, steps = make_story.check(sample())
        self.assertEqual(found, [])
        self.assertIn("Mayor Lena", steps[-1])
        self.assertIn("Pick up Cloud notebook.", steps)

    def test_normalizing_fills_in_missing_fields(self):
        story = make_story.normalized({"places": [{"id": "a", "indoor": True}], "start_place": "nowhere"})
        self.assertEqual(story["start_place"], "a")
        self.assertEqual(story["places"][0]["size"], 16.0)
        self.assertEqual(story["start_pos"], [0.0, 0.0])

    def test_reports_what_the_editor_reports(self):
        story = sample()
        story["places"][0]["doors"][0]["to"] = "moon"
        story["characters"][0]["lines"][0]["choices"][0]["next"] = 99
        story["characters"][1]["lines"][0]["choices"][0]["gives"] = "sword"
        found = make_story.problems(story)
        self.assertTrue(any("doesn't lead anywhere" in p for p in found))
        self.assertTrue(any("line index 99" in p for p in found))
        self.assertTrue(any('"sword" isn\'t an item' in p for p in found))

    def test_reports_a_story_without_an_ending(self):
        story = sample()
        for character in story["characters"]:
            for line in character["lines"]:
                for choice in line["choices"]:
                    choice["ending"] = ""
        self.assertIn("No choice ends the story.", make_story.problems(story))

    def test_reports_duplicate_ids(self):
        story = sample()
        story["characters"][1]["id"] = story["characters"][0]["id"]
        self.assertTrue(any("share the id" in p for p in make_story.problems(story)))

    def test_reports_an_ending_nobody_can_reach(self):
        story = sample()
        # Nothing sets the fact the mayor's last conversation needs any more.
        for line in make_story.find(story, "characters", "librarian")["lines"]:
            for choice in line["choices"]:
                if choice["sets"] == "learned_cycle":
                    choice["sets"] = "something_else"
        make_story.find(story, "characters", "mayor")["lines"][2]["requires"] = "something_missing"
        self.assertIsNone(make_story.walkthrough(story))
        story = sample()
        story["places"][0]["doors"] = []
        found, steps = make_story.check(story)
        self.assertIsNone(steps)
        self.assertTrue(any("no ending can be reached" in p for p in found))

    def test_a_taken_item_no_longer_counts(self):
        story = make_story.normalized({
            "places": [{"id": "yard"}],
            "start_place": "yard",
            "items": [{"id": "key", "place": "yard", "pos": [3, 3]}],
            "characters": [{"id": "guard", "place": "yard", "pos": [-3, -3], "lines": [
                {"text": "Key?", "opens": True, "choices": [
                    {"text": "Here.", "requires": "key", "takes": "key", "next": 1}]},
                {"text": "Still got it?", "choices": [
                    {"text": "Yes.", "requires": "key", "ending": "Kept it."},
                    {"text": "No.", "unless": "key", "ending": "Gave it."}]},
            ]}],
        })
        steps = make_story.walkthrough(story)
        self.assertEqual(steps, ["Pick up Thing.", 'Talk to Someone: "Here." / "No."'])

    def test_reports_things_out_of_reach(self):
        story = sample()
        mayor = make_story.find(story, "characters", "mayor")
        mayor["pos"] = [100.0, 0.0]
        house = next(p for p in story["places"][0]["props"] if p["kind"] == "house")
        story["start_pos"] = list(house["pos"])
        door = story["places"][1]["doors"][0]
        door["rot"] = 0.0  # Now it faces the wall it stands on.
        found = make_story.layout_problems(story)
        self.assertTrue(any("Mayor Lena" in p and "outside the place" in p for p in found))
        self.assertTrue(any("start point" in p and "inside the house" in p for p in found))
        self.assertTrue(any("in front of the door" in p for p in found))


class Links(unittest.TestCase):
    def test_a_link_gives_back_the_story(self):
        story = make_story.tidy(sample())
        link = make_story.link_for(story, "https://example.org/stories")
        self.assertTrue(link.startswith("https://example.org/stories/#story="))
        code = link.split("#story=")[1]
        self.assertRegex(code, r"^[A-Za-z0-9_-]+$")
        self.assertEqual(make_story.from_code(code), story)

    def test_tidy_drops_empty_text(self):
        tidied = make_story.tidy({"color": "", "pos": [0.0, 0.0], "opens": False, "lines": [{"unless": ""}]})
        self.assertEqual(tidied, {"pos": [0.0, 0.0], "opens": False, "lines": [{}]})


class Generating(unittest.TestCase):
    def test_asks_again_with_the_problems_until_the_story_passes(self):
        broken = sample()
        broken["places"][0]["doors"][0]["to"] = "moon"
        client = FakeClient([broken, sample()])
        story, found, steps = make_story.generate("the water cycle", client=client, log=lambda _: None)
        self.assertEqual(found, [])
        self.assertTrue(steps)
        self.assertEqual(story["title"], "Where Does the Water Go?")
        self.assertEqual(len(client.requests), 2)
        first, second = client.requests
        self.assertEqual(first["model"], "claude-opus-5-5")
        self.assertEqual(first["fallbacks"], "default")
        self.assertEqual(first["output_config"]["effort"], "high")
        self.assertIn("Topic: the water cycle", first["messages"][0]["content"])
        # The reply goes back whole, thinking included, then the problems.
        self.assertEqual([m["role"] for m in second["messages"]], ["user", "assistant", "user"])
        self.assertEqual(len(second["messages"][1]["content"]), 2)
        self.assertIn("doesn't lead anywhere", second["messages"][2]["content"])
        self.assertEqual(first["system"], second["system"])

    def test_gives_up_after_the_last_attempt(self):
        broken = sample()
        broken["places"][0]["doors"][0]["to"] = "moon"
        client = FakeClient([broken, broken])
        story, found, steps = make_story.generate("x", attempts=2, client=client, log=lambda _: None)
        self.assertTrue(found)
        self.assertIsNone(steps)
        self.assertEqual(len(client.requests), 2)

    def test_a_refusal_or_cut_off_reply_is_an_error(self):
        for stop_reason in ("refusal", "max_tokens"):
            with self.assertRaises(make_story.StoryError):
                make_story.generate("x", client=FakeClient([sample()], stop_reason), log=lambda _: None)

    def test_the_schema_is_strict(self):
        def walk(schema):
            if schema.get("type") == "object":
                self.assertIs(schema["additionalProperties"], False)
                self.assertEqual(schema["required"], list(schema["properties"]))
                for child in schema["properties"].values():
                    walk(child)
            elif schema.get("type") == "array":
                walk(schema["items"])

        walk(make_story.STORY_SCHEMA)


if __name__ == "__main__":
    unittest.main()
