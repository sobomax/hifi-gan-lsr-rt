#!/usr/bin/env python
# read_baseline_and_run_flake8.py

import re
import sys
from argparse import ArgumentParser
from subprocess import run

FLAKE_ARGS = (
    "flake8",
    '--format=%(code)s|%(path)s:%(row)d|%(text)s',
    '--exit-zero',
    '--ignore=I000'
)
BASELINE_FNAME = '.flake8_bsl'


def read_baseline_warnings(file_path):
    with open(file_path, 'r') as f:
        return parse_flake_warnings(f.read())


def parse_flake_warnings(output):
    pattern = re.compile(r'^[EWF][0-9][0-9][0-9]$')
    loc_pattern = re.compile(r'^\s*\^\s*$')
    last_warn = None
    res = []
    res_loc = []
    for line in output.strip().split("\n"):
        if last_warn is not None:
            last_warn.append(line)
            if len(last_warn) >= 3 and loc_pattern.match(line):
                res_loc.append(tuple(last_warn))
                code, loc, text = last_warn.pop(0)
                res.append((code,) + tuple(last_warn))
                last_warn = None
            continue
        parts = line.split('|', 3)
        if len(parts) != 3 or not pattern.match(parts[0]):
            continue
        code, loc, text = parts
        last_warn = [(code, loc, text),]
    return (tuple(res), tuple(res_loc))


def display_warning(winfo):
    print('\t' + '|'.join(winfo[0]))
    print('\t' + '\n\t'.join(winfo[1:]))


def main():
    # Parse command-line arguments
    parser = ArgumentParser(description="Check or update baseline warnings.")
    parser.add_argument('-u', '--update', action='store_true',
                        help='Update the baseline file')
    args = parser.parse_args()

    bsl_warnings = read_baseline_warnings(BASELINE_FNAME)

    # Step 1: Run flake8 as a subprocess, capturing the output
    completed_process = run(FLAKE_ARGS, capture_output=True, text=True)
    flake8_output = completed_process.stdout
    flake8_errors = completed_process.stderr

    if completed_process.returncode != 0:
        sys.stderr.write(f'{FLAKE_ARGS[0]}: command failed')
        if len(completed_process.stderr) > 0:
            sys.stderr.write(f':\n{flake8_errors}')
        else:
            sys.stderr.write('\n')
        return (4)

    # Step 3: Process this output similar to how you read your baseline
    cur_warnings = parse_flake_warnings(flake8_output)

    new_warnings = sum([1 if x not in bsl_warnings[0] else 0
                        for x in cur_warnings[0]])
    resolved_warnings = sum([1 if x not in cur_warnings[0] else 0
                             for x in bsl_warnings[0]])

    exit_status = 0
    # Optionally, you can check if new warnings have been introduced
    if new_warnings:
        nw_details = [cur_warnings[1][i] for i, x in enumerate(cur_warnings[0])
                      if x not in bsl_warnings[0]]
        print(f"New warnings: {new_warnings}")
        for w in nw_details:
            display_warning(w)
        exit_status += 1
    else:
        print("No new warnings!")

    if resolved_warnings:
        rw_details = [bsl_warnings[1][i] for i, x in enumerate(bsl_warnings[0])
                      if x not in cur_warnings[0]]
        print(f"Resolved warnings: {resolved_warnings}")
        for w in rw_details:
            display_warning(w)
        exit_status += 2

    if args.update:
        with open(BASELINE_FNAME, 'w') as f:
            f.write(flake8_output)

    return (exit_status)


if __name__ == "__main__":
    exit(main())
