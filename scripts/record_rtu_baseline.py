import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'test' / 'fixtures' / 'rtu-baseline.json'
sys.path.insert(0, str(ROOT / 'test'))
sys.path.insert(0, str(ROOT / 'src'))

from support import capture_rtu_baseline  # noqa: E402 - requires the paths above


def main():
    parser = argparse.ArgumentParser(
        description='Record the RTU implementation baseline.'
    )
    parser.add_argument(
        '--write',
        action='store_true',
        help='replace the baseline fixture with recorded values',
    )
    args = parser.parse_args()

    fixture = json.loads(FIXTURE.read_text())
    fixture['updates'] = capture_rtu_baseline(
        fixture['config'],
        fixture['seed'],
        fixture['steps'],
        fixture['output_indices'],
        fixture['state_rows'],
    )
    result = json.dumps(fixture, indent=2) + '\n'

    if args.write:
        FIXTURE.write_text(result)
    else:
        sys.stdout.write(result)


if __name__ == '__main__':
    main()
