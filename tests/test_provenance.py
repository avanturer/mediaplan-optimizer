"""Правило проекта: ни одного числа из головы.

У каждой константы в реестрах есть либо публичный источник, либо пометка
«калибровка» со ссылкой на тест, который её проверяет.
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _leaf_entries(node, path=""):
    if isinstance(node, dict):
        if "value" in node and ("status" in node or "source_url" in node or "test" in node):
            yield path, node
        else:
            for key, child in node.items():
                yield from _leaf_entries(child, f"{path}.{key}" if path else key)


def test_controller_constants_have_provenance():
    registry = yaml.safe_load((ROOT / "config" / "controller.yaml").read_text(encoding="utf-8"))
    problems = []
    for section, entries in registry.items():
        if section == "meta":
            continue
        for name, entry in entries.items():
            status = entry.get("status")
            if status == "sourced" and not entry.get("source_url"):
                problems.append(f"{section}.{name}: sourced без source_url")
            elif status == "calibrated" and not entry.get("test"):
                problems.append(f"{section}.{name}: calibrated без test")
            elif status == "numerical" and not entry.get("note"):
                problems.append(f"{section}.{name}: numerical без note (что это разрешение, а не параметр)")
            elif status not in ("sourced", "calibrated", "numerical"):
                problems.append(f"{section}.{name}: нет статуса")
    assert not problems, "\n".join(problems)


def test_calibrated_constants_point_to_existing_tests():
    registry = yaml.safe_load((ROOT / "config" / "controller.yaml").read_text(encoding="utf-8"))
    missing = []
    for section, entries in registry.items():
        if section == "meta":
            continue
        for name, entry in entries.items():
            test_ref = entry.get("test")
            if not test_ref:
                continue
            file_part, _, func = test_ref.partition("::")
            test_file = ROOT / file_part
            if not test_file.exists() or (func and func not in test_file.read_text(encoding="utf-8")):
                missing.append(f"{section}.{name} → {test_ref}")
    assert not missing, "тесты калибровки не найдены:\n" + "\n".join(missing)


def test_benchmarks_used_in_catalog_are_sourced():
    benchmarks = yaml.safe_load((ROOT / "config" / "benchmarks.yaml").read_text(encoding="utf-8"))
    unsourced = []
    for channel, fields in benchmarks["channels"].items():
        for key, node in fields.items():
            if not isinstance(node, dict) or "value" not in node:
                continue
            if node.get("value") is None:
                continue  # заглушка needs_source: в расчётах не участвует, берётся из assumptions
            if node.get("status") == "sourced" and not node.get("source_url"):
                unsourced.append(f"{channel}.{key}")
            if node.get("status") == "needs_source":
                unsourced.append(f"{channel}.{key}: помечено needs_source, но имеет значение")
    assert not unsourced, "\n".join(unsourced)
