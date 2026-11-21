#!/usr/bin/env bash
# Attended Atlas qualification only; not a production GPU lease.
set -euo pipefail
umask 077
lock=/run/infernix/gpu/atlas-7900xtx.lock
gpu=/sys/class/drm/card1/device/mem_info_vram
g=$((1 << 30))
script=$(realpath "$0")
adapter=$(dirname "$(dirname "$script")")/adapter.py

available() {
    awk '$1=="MemAvailable:" {print $2*1024; ok=1} END {if(!ok)exit 1}' OFMT=%.0f /proc/meminfo
}
psi() {
    awk '$1=="full" {sub("avg10=","",$2); print $2; ok=1} END {if(!ok)exit 1}' /proc/pressure/memory
}
below() { awk -v n="$1" -v limit="$2" 'BEGIN {exit !(n ~ /^[0-9]+(\.[0-9]+)?$/ && n<limit)}'; }
used() {
    local bytes
    bytes=$(<"${gpu}_used")
    [[ $bytes =~ ^[0-9]+$ ]] && printf '%s\n' "$bytes"
}
admit() {
    local a p total busy free
    a=$(available) && p=$(psi) && busy=$(used) && total=$(<"${gpu}_total") || return 1
    [[ $total =~ ^[0-9]+$ ]] || return 1
    free=$((total - busy))
    printf 'admission: available=%s full_psi=%s vram_free=%s\n' "$a" "$p" "$free" >&2
    ((a >= 40*g && free >= 20*g)) && below "$p" 5
}
idle() {
    [[ $(systemctl show comfyui.service -p ActiveState --value) == inactive ]] &&
        ! pgrep -f '(^|/)UnrealEditor(-Linux-Shipping)?( |$)' >/dev/null
}
holders() {
    lslocks --noheadings --notruncate -o PID,PATH | awk -v path="$lock" '$2==path {print $1}'
}

role=${1:?expected run, watch, or worker}
label=${2:?unique run label}
[[ $label =~ ^[a-z][a-z0-9-]{1,47}$ && $(hostnamectl --static) == atlas ]]
unit=infernix-qwen-"$label".service
watch=infernix-qwen-"$label"-watch.service
shift 2
run=$(realpath "${1:?private staging directory}")
python=$(realpath -s "${2:?locked venv Python}")
shift 2
cd "$run"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export ROCR_VISIBLE_DEVICES=0 HSA_OVERRIDE_GFX_VERSION=11.0.0 MIOPEN_FIND_MODE=FAST

case "$role" in
    run)
        [[ ! -e samples.tsv && ! -e result.json && ! -e ready && -s job.json ]]
        exec systemd-run --user --wait --unit="$watch" \
            -p Type=notify -p NotifyAccess=all -p WatchdogSec=5s \
            -p MemoryMax=256M -p MemorySwapMax=0 -p Restart=no -p RuntimeMaxSec=500s \
            --setenv="PATH=$PATH" --setenv="LD_LIBRARY_PATH=${LD_LIBRARY_PATH:?}" \
            --setenv="XDG_CACHE_HOME=${XDG_CACHE_HOME:?}" \
            --setenv="MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_CUSTOM_CACHE_DIR:?}" \
            -p "StandardOutput=append:$run/watch.log" -p "StandardError=append:$run/watch.log" \
            bash "$script" watch "$label" "$run" "$python" "$@"
        ;;
    worker)
        cg=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)
        [[ $(<"$cg/memory.max") == "$((24*g))" && $(<"$cg/memory.high") == "$((24*g))" &&
            $(<"$cg/memory.swap.max") == 0 ]]
        exec 9<>"${XDG_RUNTIME_DIR:?}/vagrant-gpu-atlas-7900xtx.lock"
        flock --exclusive --nonblock 9
        idle && admit
        printf '%s\n' "$cg" >ready.tmp
        mv ready.tmp ready
        for ((i=0; i<40; i++)); do
            if [[ -e go ]]; then
                idle && admit
                status=0
                "$python" -u "$adapter" --staging "$run" --backend rocm "$@" \
                    <job.json >result.json || status=$?
                printf '%s\n' "$status" >worker-status.tmp
                mv worker-status.tmp worker-status
                for ((j=0; j<30; j++)); do
                    [[ ! -e ack ]] || exit "$status"
                    sleep .1
                done
                exit 75
            fi
            sleep .1
        done
        exit 75
        ;;
    watch)
        holder=$(holders)
        admit && idle && [[ -z $holder ]] || exit 75
        systemd-notify --ready
        trap 'systemctl --user stop "$unit" || : ' EXIT
        systemd-run --user --unit="$unit" --working-directory="$run" \
            -p MemoryAccounting=yes -p MemoryHigh=24G -p MemoryMax=24G -p MemorySwapMax=0 \
            -p "BindsTo=$watch" -p "After=$watch" -p OOMPolicy=kill -p KillMode=control-group \
            -p TimeoutStopSec=1s -p Restart=no -p RuntimeMaxSec=480s -p Nice=5 \
            --setenv="PATH=$PATH" --setenv="LD_LIBRARY_PATH=$LD_LIBRARY_PATH" \
            --setenv="XDG_CACHE_HOME=$XDG_CACHE_HOME" \
            --setenv="MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_CUSTOM_CACHE_DIR" \
            -p "StandardError=append:$run/worker.log" \
            flock --shared --nonblock "$lock" bash "$script" worker "$label" "$run" "$python" "$@"
        for ((i=0; i<40; i++)); do
            [[ ! -s ready ]] || break
            sleep .1
        done
        cg=$(<ready)
        [[ $cg == /sys/fs/cgroup/*/"$unit" && -d $cg ]]
        printf 'available\tfull_psi\tvram\tpeak\tanon\tfile\tevents\n' >samples.tsv
        while [[ -d $cg ]]; do
            a=$(available)
            p=$(psi)
            vram=$(used)
            peak=$(<"$cg/memory.peak")
            read -r anon file < <(awk '$1=="anon" {a=$2} $1=="file" {f=$2} END {print a,f}' "$cg/memory.stat")
            events=$(tr '\n' ';' <"$cg/memory.events")
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$a" "$p" "$vram" "$peak" "$anon" "$file" "$events" >>samples.tsv
            if (( a < 16*g )) || ! below "$p" 10 || [[ $events =~ (high|max|oom|oom_kill|oom_group_kill)\ [1-9] ]]; then
                printf 'TRIP host/cgroup pressure: %s bytes, %s%%, %s\n' "$a" "$p" "$events" >&2
                exit 75
            fi
            holder=$(holders)
            if ! idle || [[ ! $holder =~ ^[0-9]+$ ]] || ! grep -Fq "/$unit" "/proc/$holder/cgroup"; then
                printf 'TRIP competing GPU user or missing shared lock\n' >&2
                exit 75
            fi
            if [[ -s worker-status ]]; then
                status=$(<worker-status)
                touch ack
                for ((i=0; i<20; i++)); do
                    [[ -d $cg ]] || break
                    sleep .1
                done
                holder=$(holders)
                [[ ! -d $cg && -z $holder ]] || exit 75
                printf 'after_exit_vram_bytes=%s\n' "$(used)"
                exit "$status"
            fi
            touch go
            systemd-notify WATCHDOG=1
            sleep .18
        done
        printf 'TRIP cgroup vanished before terminal counters\n' >&2
        exit 75
        ;;
    *) exit 2 ;;
esac
