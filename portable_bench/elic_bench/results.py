"""Stable JSON, flat CSV and CI-compatible JUnit outputs."""
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def summarize(records):
    counts = {name: sum(r["status"] == name for r in records)
              for name in ("passed", "failed", "error", "unavailable", "skipped")}
    completed = [r for r in records if r["status"] in ("passed", "failed")]
    checked = sum(r.get("correctness", {}).get("check_passed") is not None for r in completed)
    passed = sum(r.get("correctness", {}).get("check_passed") is True for r in completed)
    exit_code = 1 if counts["failed"] + counts["error"] else 2 if counts["unavailable"] + counts["skipped"] else 0
    return dict(status="failed" if exit_code == 1 else "partial" if exit_code == 2 else "passed",
                exit_code=exit_code, case_counts=counts,
                correctness=dict(checked_cases=checked, passed_cases=passed,
                                 pass_rate=passed/checked if checked else None,
                                 meaning="Fraction of completed correctness cases that pass; unavailable/error cases do not count as correct"),
                coverage_complete=exit_code == 0,
                cuda_execution_validated=any(r.get("cuda_execution_validated", False) and r["status"] == "passed" for r in records))


def write_outputs(report, output_dir):
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    columns = ["case_id", "suite", "status", "execution_kind", "cuda_execution_validated",
               "check_passed", "baseline_median_ms", "candidate_median_ms", "speedup", "reason"]
    with (folder / "results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for case in report["cases"]:
            row = {key: case.get(key) for key in columns[:5]}
            row["check_passed"] = case.get("correctness", {}).get("check_passed")
            timing = case.get("timing")
            if timing:
                row.update(baseline_median_ms=timing["baseline"]["median_ms"],
                           candidate_median_ms=timing["candidate"]["median_ms"], speedup=timing["speedup"])
            row["reason"] = case.get("reason", "")
            writer.writerow(row)
    counts = report["summary"]["case_counts"]
    suite = ET.Element("testsuite", name="ELiCv1 portable benchmark", tests=str(len(report["cases"])),
                       failures=str(counts["failed"]), errors=str(counts["error"]),
                       skipped=str(counts["unavailable"] + counts["skipped"]))
    for case in report["cases"]:
        element = ET.SubElement(suite, "testcase", classname=case["suite"], name=case["case_id"])
        status = case["status"]
        if status in ("failed", "error", "unavailable", "skipped"):
            tag = {"failed": "failure", "error": "error", "unavailable": "skipped", "skipped": "skipped"}[status]
            ET.SubElement(element, tag, message=case.get("reason", status)).text = case.get("traceback", "")
    ET.indent(suite)
    ET.ElementTree(suite).write(folder / "junit.xml", encoding="utf-8", xml_declaration=True)
