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
# Asight's fork interception stays off: with it the report carried only the
# first torchrun worker, and the forked DataLoader workers died as zombies.
# torchrun workers are exec'd, so the inherited environment still injects all
# eight ranks; the patched run reported one PID per device.
forks = [index for index, line in enumerate(lines) if line.startswith('PROF_FORK=')]
if len(forks) != 1:
    sys.exit("capture_report: prof.sh declares no single PROF_FORK; inspect its fork tracing first")
lines[forks[0]] = 'PROF_FORK=""\n'
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
# PCCL is deliberately not added to the trace set: the image's
# libpccl_injection.so cannot resolve its profiler symbols ("pccl injection
# subscribe fail -1"), and runs that trace pccl break the first train step
# with a SIGSEGV or a hang. Collective kernels still appear in the HGGC trace
# as pcclKernel_*.
trace = re.search(r'^([ \t]*PROF_TRACE=")(-t )([^"]*hggc[^"]*)"', patched, re.M)
if not trace:
    sys.exit("capture_report: prof.sh declares no hggc trace set; refusing to capture without it")
if ('--kill none' not in patched or 'PROF_TRACE_CAPTURE_ARGS' not in patched
        or 'trace-fork-before-exec' in patched):
    sys.exit("capture_report: patching did not produce a non-killing, fork-tracking-free, trace-selecting wrapper")
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
export PROF_TRACE_CAPTURE_ARGS="${PROF_TRACE_CAPTURE_ARGS:--t hggc,hgtx,acdnn,acblas --max-report-size ${PROF_MAX_REPORT_MB:-4096} --python-sampling false}"

exec bash "$PATCHED" "$REPORT" "$@"
