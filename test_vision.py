import base64
import importlib.util
import json
import plistlib
import tempfile
import unittest
from contextlib import redirect_stdout
from io import BytesIO, StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import fitz
from PIL import Image, ImageDraw, ImageStat

from schematic_imaging import estimate_skew, preprocess_image
import convert


spec = importlib.util.spec_from_file_location(
    "manual_server", Path(__file__).parent / "mcp-server" / "main.py",
)
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


class TileTests(unittest.TestCase):
    def test_overlap_and_remainder_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schematic.png"
            image = Image.new("L", (101, 103), 255)
            image.paste(0, (100, 102, 101, 103))
            image.save(path)
            tiles = server._crop_image_tiles(path, (2, 2), enhance=False)
            decoded = [Image.open(BytesIO(base64.b64decode(data))) for data, _ in tiles]
            self.assertEqual([tile.size for tile in decoded], [(53, 54), (54, 54), (53, 55), (54, 55)])
            self.assertLess(decoded[-1].getpixel((53, 54)), 60)

    def test_invalid_grid_and_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schematic.png"
            Image.new("L", (10, 10)).save(path)
            for grid, overlap in [((0, 2), 0.1), ((11, 2), 0.1), ((2, 2), -0.1), ((2, 2), 1)]:
                with self.subTest(grid=grid, overlap=overlap), self.assertRaises(ValueError):
                    server._crop_image_tiles(path, grid, overlap)


class ManualFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manual = self.root / "manual"
        self.manual.mkdir()
        self.schematic = self.manual / "board-schematic-p001.png"
        self.parts = self.manual / "board-parts-p002.png"
        for image in (self.schematic, self.parts):
            Image.new("L", (300, 300), 255).save(image)
        (self.manual / "01-schematic.md").write_text("# AWH-046\n![board]({})".format(self.schematic.name))
        (self.manual / "02-parts-list.md").write_text("# AWH-046\n![parts]({})".format(self.parts.name))
        self.patches = [patch.object(server, "MANUALS_DIR", self.root),
                        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key", "SCHEMATIC_MAX_HARD_CALLS": "2"}),
                        patch("sys.stderr", new=StringIO())]
        for context in self.patches:
            context.start()
            self.addCleanup(context.stop)

class CrossCheckTests(ManualFixture):
    def test_stalled_pass_escalates_and_persists_reusable_readings(self):
        existing = {"components": [{"designator": "R101", "value": "4.7k"}], "nets": [{"id": "GND"}]}
        server._save_circuit_json(self.manual, "AWH-046", existing)

        def vision(client, images, prompt, max_tokens=4096, model=None):
            if prompt.startswith("Extract component"):
                return "R101\nR102"
            return "R101\nR102" if model == server.HARD_VISION_MODEL else "R10?"

        with patch.object(server, "_vision_call", side_effect=vision) as call:
            report = server.cross_check_schematic("manual", "AWH-046")
        self.assertIn("100%", report)
        hard_calls = [entry for entry in call.call_args_list if entry.kwargs.get("model") == server.HARD_VISION_MODEL]
        self.assertEqual(len(hard_calls), 1)
        self.assertEqual(len(hard_calls[0].args[1]), 2)
        saved = server._load_circuit_json(self.manual, "AWH-046")
        self.assertEqual(saved["components"], existing["components"])
        self.assertEqual(saved["nets"], existing["nets"])
        self.assertEqual(saved["schematicReadings"]["designators"]["R101"]["status"], "confirmed")
        with patch.object(server, "_vision_call") as call, patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}):
            cached_report = server.cross_check_schematic("manual", "AWH-046")
        call.assert_not_called()
        self.assertIn("Reused 2 cached confirmed labels", cached_report)
        with patch.object(server, "_vision_call", return_value="R101\nR102") as call:
            server.cross_check_schematic("manual", "AWH-046", refresh=True)
        self.assertGreater(call.call_count, 0)

    def test_native_labels_skip_schematic_vision(self):
        metadata = {"version": 1, "image_sha256": server._source_fingerprints([self.schematic])[self.schematic.name],
                    "words": [{"text": "R101", "visible": True, "bbox": [0.1, 0.1, 0.2, 0.2]}]}
        self.schematic.with_suffix(".text.json").write_text(json.dumps(metadata))
        with patch.object(server, "_vision_call", return_value="R101") as call:
            report = server.cross_check_schematic("manual", "AWH-046")
        self.assertEqual(call.call_count, 1)
        self.assertIn("100%", report)

    def test_uncertainty_never_counts_as_confirmed_and_budget_is_enforced(self):
        def vision(client, images, prompt, max_tokens=4096, model=None):
            return "R101" if prompt.startswith("Extract component") else "R101?"
        with patch.object(server, "_vision_call", side_effect=vision) as call:
            report = server.cross_check_schematic("manual", "AWH-046")
        hard_calls = [entry for entry in call.call_args_list if entry.kwargs.get("model") == server.HARD_VISION_MODEL]
        self.assertEqual(len(hard_calls), 2)
        self.assertIn("Final coverage: **0%**", report)
        saved = server._load_circuit_json(self.manual, "AWH-046")
        self.assertEqual(saved["schematicReadings"]["designators"]["R101"]["status"], "uncertain")


    def test_partial_parts_list_is_not_reused(self):
        with patch.object(server, "_vision_call", side_effect=["R101\nR10?", "R101"]):
            report = server.cross_check_schematic("manual", "AWH-046")
        self.assertIn("coverage is provisional", report)
        with patch.object(server, "_vision_call", return_value="R101") as call:
            server.cross_check_schematic("manual", "AWH-046")
        self.assertEqual(call.call_count, 2)


class NetlistTests(ManualFixture):
    def test_netlist_reuses_labels_escalates_uncertain_connections_and_preserves_cache(self):
        with patch.object(server, "_vision_call", return_value="R101\nR102"):
            server.cross_check_schematic("manual", "AWH-046")

        def vision(client, images, prompt, max_tokens=4096, model=None):
            if model == server.HARD_VISION_MODEL:
                return "R102 p1_n1 GND 100"
            if prompt.startswith("You are an expert"):
                self.assertIn("Prior label readings", prompt)
                return "R101 VCC n1 4700\nR102 n1? GND 100"
            return "R102 n9 GND 100"

        with patch.object(server, "_vision_call", side_effect=vision) as call:
            report = server.generate_netlist("manual", "AWH-046")
        self.assertIn("Final coverage: **100%**", report)
        hard_calls = [entry for entry in call.call_args_list if entry.kwargs.get("model") == server.HARD_VISION_MODEL]
        self.assertEqual(len(hard_calls), 1)
        self.assertEqual(len(hard_calls[0].args[1]), 3)
        saved = server._load_circuit_json(self.manual, "AWH-046")
        self.assertEqual(len(saved["components"]), 2)
        self.assertIn("schematicReadings", saved)
        shared = next(net for net in saved["nets"] if net["id"] == "p1_n1")
        self.assertEqual(len(shared["connectedPins"]), 2)
        self.assertTrue(all(pin["netID"] for component in saved["components"] for pin in component["pins"]))

    def test_uncertain_connections_do_not_count_or_save(self):
        entries, netted = server._parse_spice_lines("R101 n1? GND 100\nR102 n2 GND 200?")
        self.assertFalse(netted)
        self.assertFalse(server._resolved_designators(entries))
        with patch.object(server, "_vision_call", return_value="R101 n1? GND 100"):
            report = server.generate_netlist("manual", "AWH-046")
        self.assertIn("* UNRESOLVED:", report)
        self.assertIsNone(server._load_circuit_json(self.manual, "AWH-046"))

    def test_pin_names_override_component_array_order(self):
        circuit = {"components": [{"designator": "Q101", "pins": [{"id": "B"}, {"id": "C"}, {"id": "E"}]}]}
        entries, _ = server._parse_spice_lines("Q101 VCC IN GND NPN")
        updated = server._spice_to_circuit_json(entries, circuit)
        self.assertEqual({pin["id"]: pin["netID"] for pin in updated["components"][0]["pins"]},
                         {"C": "VCC", "B": "IN", "E": "GND"})
        self.assertNotIn("netID", circuit["components"][0]["pins"][0])

    def test_save_false_leaves_existing_json_unchanged(self):
        path = server._save_circuit_json(self.manual, "AWH-046", {"components": [], "nets": [], "custom": "keep"})
        before = path.read_bytes()
        with patch.object(server, "_vision_call", return_value="R101 VCC GND 100"):
            report = server.generate_netlist("manual", "AWH-046", save_json=False)
        self.assertIn("R101 VCC GND 100", report)
        self.assertEqual(path.read_bytes(), before)


class RetryAndCacheTests(unittest.TestCase):
    def test_hard_budget_can_be_disabled_or_bounded(self):
        for value, expected in [("0", 0), ("-1", 0), ("50", 20), ("invalid", 4)]:
            with self.subTest(value=value), patch.dict("os.environ", {"SCHEMATIC_MAX_HARD_CALLS": value}):
                self.assertEqual(server._hard_call_budget(), expected)

    def test_model_override_and_truncated_responses(self):
        client = Mock()
        client.messages.create.return_value = SimpleNamespace(
            stop_reason="end_turn", content=[SimpleNamespace(type="text", text="R101")],
        )
        self.assertEqual(server._vision_call(client, [], "read", model="hard-model"), "R101")
        self.assertEqual(client.messages.create.call_args.kwargs["model"], "hard-model")
        client.messages.create.return_value.stop_reason = "max_tokens"
        with self.assertRaises(ValueError):
            server._vision_call(client, [], "read")

    def test_cache_invalidates_changed_sources_and_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schematic.png"
            Image.new("L", (100, 100), 255).save(path)
            cache = {
                "version": 1, "schematicImages": [path.name], "settings": server._reading_settings(),
                "sources": server._source_fingerprints([path]), "designators": {"R101": {"status": "confirmed"}},
            }
            circuit = {"components": [], "schematicReadings": cache}
            self.assertEqual(server._cached_readings(circuit, [path]), cache)
            with patch.object(server, "VISION_MODEL", "different"):
                self.assertEqual(server._cached_readings(circuit, [path]), {})
            Image.new("L", (100, 100), 0).save(path)
            self.assertEqual(server._cached_readings(circuit, [path]), {})

    def test_crops_target_cached_locations_and_keep_raw_variant(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schematic.png"
            Image.new("L", (300, 300), 255).save(path)
            readings = {"R101": {"evidence": [{"image": path.name, "bbox": [0.1, 0.1, 0.2, 0.2]}]}}
            tiles = list(server._refinement_tiles(path, (3, 3), {"R101"}, readings, hard=True))
            self.assertEqual(len(tiles), 1)
            self.assertEqual(tiles[0][0], 0)
            self.assertEqual(len(tiles[0][1]), 2)


class TextEvidenceTests(unittest.TestCase):
    def test_convert_mixed_manual_to_default_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "Mixed Manual.pdf"
            with fitz.open() as document:
                page = document.new_page(width=300, height=200)
                page.insert_text((20, 40), "R101 4.7k C102 100u")
                page.insert_text((20, 60), "Power Amplifier Assembly (AWH-046)")
                page = document.new_page(width=300, height=200)
                image = Image.new("L", (300, 200), 210)
                drawing = ImageDraw.Draw(image)
                drawing.line((20, 100, 280, 100), fill=160, width=2)
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                page.insert_image(page.rect, stream=buffer.getvalue())
                document.set_toc([[1, "Power Amplifier Assembly (AWH-046)", 1],
                                  [1, "Parts List (AWH-046)", 2]])
                document.save(pdf)
            with patch.object(convert, "DEFAULT_OUTPUT_DIR", root / "Schematics"), redirect_stdout(StringIO()):
                convert.convert_pdf(str(pdf))
            output = root / "Schematics" / "mixed-manual"
            self.assertTrue((output / "_index.md").exists())
            self.assertEqual(len(list(output.glob("*.png"))), 2)
            metadata = [json.loads(path.read_text()) for path in sorted(output.glob("*.text.json"))]
            self.assertEqual(len(metadata), 2)
            self.assertEqual(sorted(bool(item["words"]) for item in metadata), [False, True])

    def test_text_schematic_keeps_image_and_positioned_labels(self):
        with tempfile.TemporaryDirectory() as directory, fitz.open() as document:
            page = document.new_page(width=300, height=200)
            page.insert_text((20, 40), "R101 4.7k C102 100u POWER AMPLIFIER AWH-046")
            page.insert_text((20, 80), "R999", render_mode=3)
            section = {
                "title": "Power Amplifier Assembly (AWH-046)", "pages": {0},
                "text_parts": [(0, page.get_text())],
            }
            output = Path(directory)
            result = convert.write_section(section, 1, output, ["text"], document, "schematic")
            self.assertEqual(len(result[-1]), 1)
            image_path = output / result[-1][0]
            evidence = server._native_designators(image_path)
            self.assertIn("R101", evidence)
            self.assertIn("C102", evidence)
            self.assertNotIn("R999", evidence)
            self.assertTrue(all(0 <= coord <= 1 for coord in evidence["R101"][0]["bbox"]))
            self.assertIn("R999", server._page_text_context(image_path))
            image_path.write_bytes(b"changed image")
            self.assertEqual(server._native_designators(image_path), {})

    def test_legacy_images_work_without_sidecars(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(server._native_designators(Path(directory) / "legacy.png"), {})


class StorageTests(unittest.TestCase):
    def test_entry_points_share_storage_root(self):
        expected = Path("/Users/marshallbenson/Desktop/Benchmark Audio Repair/Schematics")
        self.assertEqual(server.MANUALS_DIR, expected)
        self.assertEqual(server.SCHEMATICS_DIR, expected)
        self.assertEqual(convert.DEFAULT_OUTPUT_DIR, expected)
        workflow_path = Path(__file__).parent / "Extract for Claude.workflow/Contents/document.wflow"
        with workflow_path.open("rb") as stream:
            workflow = plistlib.load(stream)
        command = workflow["actions"][0]["action"]["ActionParameters"]["COMMAND_STRING"]
        self.assertIn('OUTPUT_DIR="{}"'.format(expected), command)
        self.assertNotIn("Claude-Manuals", command)


class PreprocessingTests(unittest.TestCase):
    def test_deskew_low_contrast_scan_without_mutating_original(self):
        image = Image.new("L", (600, 400), 210)
        drawing = ImageDraw.Draw(image)
        for position in range(40, 380, 30):
            drawing.line((50, position, 550, position), fill=160, width=2)
        tilted = image.rotate(2, expand=True, fillcolor=210)
        before = tilted.tobytes()
        self.assertAlmostEqual(estimate_skew(tilted), -2, delta=0.3)
        enhanced = preprocess_image(tilted)
        self.assertLess(abs(estimate_skew(enhanced)), 0.3)
        self.assertGreater(ImageStat.Stat(enhanced).stddev[0], ImageStat.Stat(tilted).stddev[0])
        self.assertEqual(tilted.tobytes(), before)

    def test_blank_and_disabled_preprocessing(self):
        image = Image.new("L", (80, 60), 210)
        self.assertEqual(estimate_skew(image), 0)
        self.assertEqual(preprocess_image(image).size, image.size)
        with patch.dict("os.environ", {"SCHEMATIC_PREPROCESS": "0"}):
            self.assertEqual(preprocess_image(image).tobytes(), image.tobytes())


if __name__ == "__main__":
    unittest.main()