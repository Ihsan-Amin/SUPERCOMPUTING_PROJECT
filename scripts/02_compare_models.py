#!/usr/bin/env python3
"""
Compare test results from all three Fruits-262 models.
Reads test_results.json from each model subdirectory and prints a
formatted comparison table including paper benchmarks.
"""

import os
import json
import argparse
#TEST OUTPUT CHANGE
import shutil

DEFAULT_OUTPUT_DIR = "/sciclone/scr10/gzdata440/fruitsdata2/output"
#mark for later

# Paper benchmarks from Table VIII (52x64 RGB model)
PAPER_BENCHMARKS = {
    "model": "paper (52x64 RGB)",
    "resolution": "52x64",
    "test_top1": 59.15,
    "test_top5": 80.40,
    "test_top10": 86.66,
    "total_params": "~5M",
    "training_hours": "N/A",
}


def main():
    parser = argparse.ArgumentParser(description="Compare Fruits-262 model results")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    #TEST OUTPUT CHANGE
    parser.add_argument("--report-dir", type=str, default=None)
    args = parser.parse_args()

    models = ["alexnet", "alexnet_bn", "resnet50"]
    results = []

    for model_name in models:
        path = os.path.join(args.output_dir, model_name, "test_results.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            results.append(data)
        else:
            print(f"  WARNING: {path} not found, skipping {model_name}")

    if not results:
        print("No results found. Have the models finished training?")
        return

    # ── Print comparison table ───────────────────────────────────────────
    print()
    print("=" * 90)
    print("  FRUITS-262 MODEL COMPARISON")
    print("=" * 90)
    print()

    header = (f"{'Model':<18} {'Resolution':<12} {'Top-1':>7} {'Top-5':>7} "
              f"{'Top-10':>7} {'Params':>12} {'Hours':>7} {'Optimizer':<10}")
    print(header)
    print("-" * 90)

    # Paper benchmark row
    p = PAPER_BENCHMARKS
    print(f"{'paper (Table VIII)':<18} {'52x64':<12} {p['test_top1']:>6.2f}% "
          f"{p['test_top5']:>6.2f}% {p['test_top10']:>6.2f}% "
          f"{'~5M':>12} {'N/A':>7} {'Adam':<10}")
    print("-" * 90)

    # Our models
    for r in results:
        params_str = f"{r['total_params']:,}" if isinstance(r['total_params'], int) else str(r['total_params'])
        hours_str = f"{r['training_hours']:.2f}" if isinstance(r['training_hours'], (int, float)) else str(r['training_hours'])
        opt = r.get('optimizer', 'N/A')

        print(f"{r['model']:<18} {r['resolution']:<12} {r['test_top1']:>6.2f}% "
              f"{r['test_top5']:>6.2f}% {r['test_top10']:>6.2f}% "
              f"{params_str:>12} {hours_str:>7} {opt:<10}")

    print("-" * 90)
    print()

    # ── Highlights ───────────────────────────────────────────────────────
    if len(results) >= 2:
        best = max(results, key=lambda r: r["test_top1"])
        worst = min(results, key=lambda r: r["test_top1"])

        print(f"  Best model:   {best['model']} ({best['test_top1']:.2f}% top-1)")
        print(f"  Worst model:  {worst['model']} ({worst['test_top1']:.2f}% top-1)")

        # Compare to paper
        paper_top1 = PAPER_BENCHMARKS["test_top1"]
        for r in results:
            delta = r["test_top1"] - paper_top1
            direction = "above" if delta > 0 else "below"
            print(f"  {r['model']}: {abs(delta):.2f}% {direction} paper benchmark")

    print()

    # ── Save comparison CSV ──────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, "model_comparison.csv")
    with open(csv_path, "w") as f:
        f.write("model,resolution,test_top1,test_top5,test_top10,"
                "params,hours,optimizer,scheduler,epochs,batch_size\n")
        for r in results:
            f.write(f"{r['model']},{r['resolution']},{r['test_top1']:.4f},"
                    f"{r['test_top5']:.4f},{r['test_top10']:.4f},"
                    f"{r['total_params']},{r['training_hours']:.4f},"
                    f"{r.get('optimizer','')},{r.get('scheduler','')},{r['epochs_trained']},"
                    f"{r['batch_size']}\n")

    print(f"  Comparison CSV saved to: {csv_path}")

    #TEST OUTPUT CHANGE
    if args.report_dir:
        os.makedirs(args.report_dir, exist_ok=True)
        shutil.copy2(csv_path, os.path.join(args.report_dir, "model_comparison.csv"))
        print(f"  Also copied to: {args.report_dir}")

    print()


if __name__ == "__main__":
    main()
