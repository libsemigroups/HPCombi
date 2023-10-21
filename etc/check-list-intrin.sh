#!/bin/bash
set -e

bold() {
  printf "\033[1m%s\033[0m\n" "$*"
}

if [[ $# -ne 0 ]]; then
  bold "error expected no arguments, got $#!"
  exit 1
fi

echo "Checking intrinsics used in code base versus declared in 'list_intrin.txt' . . ."

grep -oh -e "_mm\w*\b" include/*.* | sort | uniq > intrin_actuals.txt

python3 - <<END
import sys
with open('intrin_actuals.txt', 'r') as f:
  intrin_actuals = [x.strip() for x in f.readlines()]
with open('list_intrin.txt', 'r') as f:
  list_intrin = [x.strip()[:x.find(';')] for x in f.readlines() if not x.startswith('#')]

for intrin_actual in intrin_actuals:
  if not intrin_actual in list_intrin:
    print(f"\033[41m{('\`' + intrin_actual + '\`').ljust(16, ' ')} found in \`include/*.hpp\`   but not in \`list_intrin.txt\`!\033[0m")

for intrin in list_intrin:
  if not intrin in intrin_actuals:
    print(f"\033[41m{('\`' + intrin + '\`').ljust(16, ' ')} found in \`list_intrin.txt\` but not in \`include/*.hpp\`!\033[0m")
END

rm -f intrin_actuals.txt
