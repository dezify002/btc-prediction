#!/usr/bin/env python3
"""
Phase 4 — Learning AI CLI
Run analyses and generate reports from the command line.

Usage:
    cd models
    python phase4_cli.py --summary          # Quick daily summary
    python phase4_cli.py --weekly            # Full weekly report
    python phase4_cli.py --drift              # Drift detection only
    python phase4_cli.py --patterns           # Pattern discovery
    python phase4_cli.py --calibrate 75       # Calibrate confidence=75
    python phase4_cli.py --recommend          # Top recommendations
    python phase4_cli.py --all                # Run everything
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from learning_engine import LearningEngine


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_accuracy_bar(accuracy: float, width: int = 30):
    filled = int((accuracy / 100) * width)
    bar = "█" * filled + "░" * (width - filled)
    print(f"  [{bar}] {accuracy:.1f}%")


def run_summary(engine: LearningEngine):
    print_header("DAILY SUMMARY")
    summary = engine.daily_summary()

    if "error" in summary:
        print(f"  ⚠️  {summary['error']}")
        return

    ov = summary["overall"]
    print(f"  Total Predictions: {ov['total_predictions']:,}")
    print(f"  Overall Accuracy:  {ov['accuracy_pct']}%")
    print_accuracy_bar(ov['accuracy_pct'])

    if ov.get("last_7_days_accuracy"):
        print(f"  Last 7 Days:       {ov['last_7_days_accuracy']}%")
    if ov.get("last_30_days_accuracy"):
        print(f"  Last 30 Days:      {ov['last_30_days_accuracy']}%")

    print(f"\n  Today: {summary['today']['predictions']} predictions, "
          f"{summary['today']['correct']} correct ({summary['today']['accuracy']}%)")


def run_weekly(engine: LearningEngine):
    print_header("WEEKLY REPORT")
    report = engine.weekly_report()

    if "error" in report:
        print(f"  ⚠️  {report['error']}")
        return

    s = report["summary"]
    print(f"  Period: {report['period']['from'][:10]} → {report['period']['to'][:10]}")
    print(f"  Predictions: {s['total_predictions']:,}")
    print(f"  Correct:     {s['correct']:,}")
    print(f"  Accuracy:    {s['accuracy_pct']}%")
    print_accuracy_bar(s['accuracy_pct'])

    print(f"\n  Best Window:   {report['best_window']['name']} ({report['best_window'].get('accuracy_pct', 'N/A')}%)")
    print(f"  Worst Window:  {report['worst_window']['name']} ({report['worst_window'].get('accuracy_pct', 'N/A')}%)")
    print(f"  Best Regime:   {report['best_regime']['name']} ({report['best_regime'].get('accuracy_pct', 'N/A')}%)")
    print(f"  Worst Regime:  {report['worst_regime']['name']} ({report['worst_regime'].get('accuracy_pct', 'N/A')}%)")
    print(f"  Most Reliable Confidence: {report['most_reliable_confidence']}")


def run_drift(engine: LearningEngine):
    print_header("DRIFT DETECTION")
    drift = engine.detect_drift()

    if "error" in drift:
        print(f"  ⚠️  {drift['error']}")
        return

    status_icons = {
        "stable": "✅",
        "improving": "📈",
        "warning": "⚠️",
        "declining": "🚨"
    }
    icon = status_icons.get(drift["status"], "❓")

    print(f"  {icon} Status: {drift['status'].upper()}")
    print(f"  Baseline Accuracy: {drift['baseline_accuracy']}% ({drift['baseline_window']} preds)")
    print(f"  Recent Accuracy:   {drift['recent_accuracy']}% ({drift['recent_window']} preds)")
    print(f"  Drift:             {drift['drift_pct']:+.1f}%")
    print(f"  P-value:           {drift['p_value']:.4f} {'(significant)' if drift['significant'] else '(not significant)'}")
    print(f"\n  💡 {drift['recommendation']}")


def run_patterns(engine: LearningEngine):
    print_header("PATTERN DISCOVERY")
    patterns = engine.discover_patterns()

    if not patterns or "error" in patterns[0]:
        print(f"  ⚠️  {patterns[0].get('error', 'No patterns found')}")
        return

    for p in patterns:
        impact_icon = "🔴" if p.get("impact") == "high" else "🟡" if p.get("impact") == "medium" else "🟢"
        print(f"\n  {impact_icon} [{p['category'].upper()}]")
        print(f"     Finding: {p['finding']}")
        print(f"     → {p.get('recommendation', 'No recommendation')}")


def run_calibration(engine: LearningEngine, confidence: float = None):
    print_header("CONFIDENCE CALIBRATION")
    cal = engine.calibrate_confidence()

    if "error" in cal:
        print(f"  ⚠️  {cal['error']}")
        return

    print(f"  Brier Score: {cal['brier_score']}")
    print(f"  Assessment:  {cal['overall_assessment']}")
    print(f"\n  {'Bin':<15} {'Count':>8} {'Avg Conf':>10} {'Actual':>10} {'Gap':>10} {'Status':<15}")
    print(f"  {'-'*15} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*15}")

    for b in cal["bins"]:
        gap = b["calibration_gap"]
        gap_str = f"+{gap:.1f}" if gap > 0 else f"{gap:.1f}"
        print(f"  {b['bin_label']:<15} {b['count']:>8} {b['avg_confidence']:>9.1f}% "
              f"{b['actual_accuracy']:>9.1f}% {gap_str:>9}% {b['reliability']:<15}")

    if confidence is not None:
        result = engine.get_calibrated_confidence(confidence)
        print(f"\n  📊 Calibrated Confidence for {confidence}%:")
        print(f"     Historical Accuracy: {result['calibrated_confidence']}%")
        print(f"     Based on: {result['sample_size']} similar predictions")
        print(f"     Method: {result['method']}")


def run_recommendations(engine: LearningEngine):
    print_header("TOP RECOMMENDATIONS")
    recs = engine.generate_recommendations()

    if not recs:
        print("  ✅ No critical recommendations. Model is performing well.")
        return

    priority_icons = {"critical": "🚨", "high": "🔴", "medium": "🟡", "low": "🟢"}

    for i, rec in enumerate(recs[:10], 1):
        icon = priority_icons.get(rec["priority"], "⚪")
        print(f"\n  {icon} #{i} [{rec['priority'].upper()}] {rec['title']}")
        print(f"     {rec['description']}")
        print(f"     → Action: {rec['action']}")


def run_all(engine: LearningEngine):
    run_summary(engine)
    run_weekly(engine)
    run_drift(engine)
    run_patterns(engine)
    run_calibration(engine)
    run_recommendations(engine)

    # Save full report
    report = engine.weekly_report()
    path = engine.save_report(report, f"weekly_report_{engine.artifacts_dir.name}.json")
    print(f"\n📄 Full report saved to: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4 — Learning AI CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python phase4_cli.py --summary
  python phase4_cli.py --weekly
  python phase4_cli.py --calibrate 82 --regime uptrend_low_vol
  python phase4_cli.py --all
        """
    )

    parser.add_argument("--summary", action="store_true", help="Daily summary")
    parser.add_argument("--weekly", action="store_true", help="Weekly report")
    parser.add_argument("--drift", action="store_true", help="Drift detection")
    parser.add_argument("--patterns", action="store_true", help="Pattern discovery")
    parser.add_argument("--calibrate", type=float, metavar="CONF", help="Calibrate a confidence value")
    parser.add_argument("--regime", type=str, help="Market regime for calibration")
    parser.add_argument("--window", type=str, help="Prediction window for calibration")
    parser.add_argument("--recommend", action="store_true", help="Top recommendations")
    parser.add_argument("--all", action="store_true", help="Run all analyses")
    parser.add_argument("--log-dir", type=str, default="../reports/logs", help="Log directory (default: ../reports/logs)")
    parser.add_argument("--artifacts", type=str, default="../reports", help="Artifacts directory")
    parser.add_argument("--days", type=int, help="Only analyze last N days")

    args = parser.parse_args()

    if not any([args.summary, args.weekly, args.drift, args.patterns, 
                args.calibrate is not None, args.recommend, args.all]):
        parser.print_help()
        sys.exit(1)

    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    log_dir = script_dir / args.log_dir
    artifacts_dir = script_dir / args.artifacts

    # Initialize engine
    engine = LearningEngine(log_dir=str(log_dir), artifacts_dir=str(artifacts_dir))
    engine.load_logs(days=args.days)

    if not engine.has_data(5):
        print("⚠️  No prediction logs found.")
        print(f"   Looking in: {log_dir.absolute()}")
        print("   Run some predictions first or check your log path.")
        sys.exit(1)

    print(f"📊 Loaded {len(engine.df)} predictions from {log_dir}")

    if args.summary:
        run_summary(engine)
    if args.weekly:
        run_weekly(engine)
    if args.drift:
        run_drift(engine)
    if args.patterns:
        run_patterns(engine)
    if args.calibrate is not None:
        run_calibration(engine, args.calibrate)
    if args.recommend:
        run_recommendations(engine)
    if args.all:
        run_all(engine)


if __name__ == "__main__":
    main()