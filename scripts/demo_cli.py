"""Interactive / single-command CLI demonstrating ClinGuard-RAG decision stages and audit records."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from clinguard_rag.pipeline import ClinGuardPipeline
from clinguard_rag.retriever import HybridClinicalRetriever

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="ClinGuard-RAG Interactive Demo CLI")
    parser.add_argument("--case-id", type=str, default="EVAL-03", help="Benchmark case ID to inspect")
    parser.add_argument(
        "--mode",
        type=str,
        default="clinguard_full",
        choices=["baseline", "raw_system1", "debiased_system1", "clinguard_full"],
    )
    args = parser.parse_args()

    retriever = HybridClinicalRetriever.from_json(ROOT / "data" / "clinical_guidelines_corpus.json")
    cases = json.loads((ROOT / "data" / "evaluation_benchmark.json").read_text(encoding="utf-8"))
    cal_cases = [c for c in cases if c["split"] == "calibration"]

    pipeline = ClinGuardPipeline(retriever=retriever, mode=args.mode)
    if args.mode == "clinguard_full":
        pipeline.fit_calibration(cal_cases)

    target = next((c for c in cases if c["case_id"] == args.case_id), cases[0])
    audit = pipeline.run_case(target)
    print(json.dumps(audit.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
