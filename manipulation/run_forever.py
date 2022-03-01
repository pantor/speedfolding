from pathlib import Path
import subprocess
import sys

# Import argument parser from run.py file
from run import parser, logger


if __name__ == '__main__':
    args = parser.parse_args()

    script_path = Path(__file__).parent.absolute() / 'run.py'
    cmd = ['python3', str(script_path)]
    cmd += sys.argv[1:]  # Add optional arguments and flags
    logger.info(f"Executing: {' '.join(cmd)}")

    for step in range(1_000_000_000):
        logger.info(f'Starting round {step} of data collection')

        # Add reset-at-startup for all but the first run
        proc = subprocess.Popen(cmd + (['--reset-at-startup', '--eval-save', '--eval-load'] if step > 0 else ['--eval-save']))
        proc.wait()
