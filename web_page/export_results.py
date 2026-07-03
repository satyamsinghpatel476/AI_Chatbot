from __future__ import annotations

import sys

from data_adapter import DEFAULT_OUTPUT_PATH, DEFAULT_RESULTS_PATH, export_summary


def main() -> int:
    try:
        output_path, dashboard_data = export_summary(DEFAULT_OUTPUT_PATH, DEFAULT_RESULTS_PATH)
    except ValueError as exc:
        print("Validation failed. Could not export webpage data.")
        for line in str(exc).splitlines():
            print(f"- {line}")
        return 1
    except OSError as exc:
        print("Export failed while writing results_summary.json.")
        print(f"- {exc}")
        return 1

    print("Export complete.")
    print(f"- Source: {dashboard_data['source']['path']}")
    print(f"- Source modified: {dashboard_data['source']['modified_at'] or 'N/A'}")
    print(f"- Output: {output_path}")
    print(f"- Questions exported: {dashboard_data['summary_cards']['total_questions']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
