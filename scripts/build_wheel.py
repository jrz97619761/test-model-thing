"""Build a wheel and remove temporary Setuptools artifacts."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


def clean(root):
    for path in [root / 'build', *sorted((root / 'src').glob('*.egg-info'))]:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clean-only', action='store_true',
                        help='remove build/ and src/*.egg-info without a build')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        if not args.clean_only:
            subprocess.run([sys.executable, '-m', 'build', '--wheel', '--outdir', str(root / 'dist')],
                           cwd=root, check=True)
    finally:
        clean(root)


if __name__ == '__main__':
    main()
