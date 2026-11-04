#!/usr/bin/env bash
#
# Run one entrypoint under model_prof's prof.sh and keep the captured report.
#
#   bash capture_report.sh <report-prefix> <command...>
#
# The wrapper is fail-closed: if prof.sh cannot be found or its command layout
# changed, the run stops instead of silently training without a capture.
#
# prof.sh from model_prof 0.1.33 stops the target when the capture window closes
# and does not inject the trace selection. A CI run needs both steps to finish,
# so this copies prof.sh to a new file next to the report and changes those two
# behaviors there. The installed prof.sh is never modified.
set -uo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <report-prefix> <command...>" >&2
    exit 2
fi

REPORT="$1"
shift

SOURCE="${MODEL_PROF_SH:-}"
if [[ -z "$SOURCE" ]]; then
    PKG_DIR="$(python -c 'import model_prof, os; print(os.path.dirname(model_prof.__file__))' 2>/dev/null)" || {
        echo "capture_report: model_prof is not importable; set MODEL_PROF_SH to its prof.sh" >&2
        exit 3
    }
    for candidate in "$PKG_DIR/tool/prof.sh" "$PKG_DIR/prof.sh"; do
        [[ -f "$candidate" ]] && SOURCE="$candidate" && break
    done
fi
[[ -f "$SOURCE" ]] || { echo "capture_report: prof.sh not found; set MODEL_PROF_SH" >&2; exit 3; }

WORK_DIR="$(dirname "$REPORT")"
mkdir -p "$WORK_DIR"
PATCHED="$WORK_DIR/prof_capture_$(basename "$REPORT").sh"
[[ -f "$PATCHED" || -L "$PATCHED" ]] && { echo "capture_report: refusing to overwrite $PATCHED" >&2; exit 4; }

python3 - "$SOURCE" "$PATCHED" <<'PY'
import pathlib
import re
import sys

source, output = (pathlib.Path(value) for value in sys.argv[1:3])
lines = source.read_text().splitlines(keepends=True)
commands = [index for index, line in enumerate(lines) if 'PROF_CMD="${PROF_TOOL} profile' in line]
if len(commands) != 2:
    sys.exit("capture_report: unsupported prof.sh layout; inspect its profile command branches first")
for index in commands:
    line = lines[index]
    if '${RAY_CMD} ${CMD_STR}' not in line:
        sys.exit("capture_report: prof.sh command no longer matches the expected form")
    line = line.replace('--kill 9', '--kill none')
    if 'PROF_TRACE_CAPTURE_ARGS' not in line:
        line = line.replace('${RAY_CMD} ${CMD_STR}', '${RAY_CMD} ${PROF_TRACE_CAPTURE_ARGS} ${CMD_STR}')
    lines[index] = line
patched = ''.join(lines)
# The default trace set omits PCCL; a cluster run needs the collective
# kernels in the report as well. The default set differs between prof.sh
# versions, so extend whatever hggc list is declared instead of matching it
# verbatim.
trace = re.search(r'^([ \t]*PROF_TRACE=")(-t )([^"]*hggc[^"]*)"', patched, re.M)
if not trace:
    sys.exit("capture_report: prof.sh declares no hggc trace set; refusing to capture without it")
if 'pccl' not in [item.strip() for item in trace.group(3).split(',')]:
    patched = patched[:trace.start(3)] + trace.group(3) + ',pccl' + patched[trace.end(3):]
if '--kill none' not in patched or 'PROF_TRACE_CAPTURE_ARGS' not in patched or 'pccl' not in patched:
    sys.exit("capture_report: patching did not produce a non-killing, trace-selecting wrapper")
with output.open('x') as stream:
    stream.write(patched)
PY
status=$?
[[ $status -eq 0 ]] || exit $status
bash -n "$PATCHED" || exit 5

# MODEL_PROFILE makes prof_iter() open and close the capture API range; the
# range is a closed interval, so 10,11 captures exactly two complete steps.
export MODEL_PROFILE=1
export MODEL_PROFILE_RANGE="${MODEL_PROFILE_RANGE:-10,11}"
unset MODEL_PROFILE_DISABLE_PROF_RANGE
export PROF_DELAY=0
export PROFILE_DELAY=0
export PROF_SKIP_SUMMARY_GEN=1
export PROF_TRACE_CAPTURE_ARGS="${PROF_TRACE_CAPTURE_ARGS:--t hggc,hgtx,acdnn,acblas,pccl --max-report-size ${PROF_MAX_REPORT_MB:-4096} --python-sampling false}"

exec bash "$PATCHED" "$REPORT" "$@"
