#!/bin/sh
#
# disable_acs_p2p.sh - clear the P2P-poisoning ACS Control bits
# (ReqRedir, CmpltRedir, UpstreamFwd) on given PCI bridges, via the ACS
# extended capability register. Leaves SrcValid/TransBlk/EgressCtrl/
# DirectTrans untouched. Run as root (sudo).
#
# Companion to acs.sh: run acs.sh first to see which bridges show
# ReqRedir+/CmpltRedir+/UpstreamFwd+ above a GPU, then pass those BDFs here.
#
# Usage: sudo ./disable_acs_p2p.sh <bdf> [<bdf> ...]
#   e.g. sudo ./disable_acs_p2p.sh 0a:00.0 0d:00.0
#
# This is a runtime fix only - BIOS/firmware or a PCIe hot-reset can
# silently re-arm these bits, and a reboot restores BIOS defaults. Verify
# after running with: sudo ./acs.sh
#
# Copyright Kevin Read <me@kevin-read.com>

set -eu

if [ "$#" -eq 0 ]; then
    echo "usage: $0 <bdf> [<bdf> ...]" >&2
    echo "  e.g.  $0 0a:00.0 0d:00.0" >&2
    exit 1
fi

REQ_REDIR_BIT=2
CMPLT_REDIR_BIT=3
UPSTREAM_FWD_BIT=4

clear_acs_named() {
    bdf="$1"
    ctl_before=$(setpci -s "$bdf" ECAP_ACS+6.w 2>/dev/null) || return 1
    [ -z "$ctl_before" ] && return 1
    ctl_val=$((16#$ctl_before))
    ctl_val=$(( ctl_val & ~(1 << REQ_REDIR_BIT) & ~(1 << CMPLT_REDIR_BIT) & ~(1 << UPSTREAM_FWD_BIT) ))
    ctl_after=$(printf '%04x' "$ctl_val")
    echo "  ACSCtl (via ECAP_ACS): before=$ctl_before after=$ctl_after"
    setpci -s "$bdf" "ECAP_ACS+6.w=${ctl_after}"
}

clear_acs_walk() {
    # Fallback for setpci builds without ECAP_ACS symbolic support: walk the
    # PCIe extended capability list (starts at 0x100) to find ACS
    # (cap ID 0x000D), then ACSCtl is at cap_offset+6.
    bdf="$1"
    offset=256  # 0x100
    while :; do
        off_hex=$(printf '0x%x' "$offset")
        header=$(setpci -s "$bdf" "${off_hex}.l" 2>/dev/null) || return 1
        [ -z "$header" ] && return 1
        capid=$((16#${header:6:4}))
        nextptr=$(( (16#${header:2:3}) & 0xFFC ))
        if [ "$capid" = "13" ]; then   # 0x000D ACS extended cap
            ctl_off=$(printf '0x%x' $((offset + 6)))
            ctl_before=$(setpci -s "$bdf" "${ctl_off}.w")
            ctl_val=$((16#$ctl_before))
            ctl_val=$(( ctl_val & ~(1 << REQ_REDIR_BIT) & ~(1 << CMPLT_REDIR_BIT) & ~(1 << UPSTREAM_FWD_BIT) ))
            ctl_after=$(printf '%04x' "$ctl_val")
            echo "  ACS cap at offset $off_hex, ctl reg at $ctl_off"
            echo "  ACSCtl (via ext-cap walk): before=$ctl_before after=$ctl_after"
            setpci -s "$bdf" "${ctl_off}.w=${ctl_after}"
            return 0
        fi
        [ "$nextptr" = "0" ] && return 1
        offset=$nextptr
    done
}

for bdf in "$@"; do
    echo "=== $bdf ==="
    if clear_acs_named "$bdf" 2>/dev/null; then
        :
    elif clear_acs_walk "$bdf"; then
        :
    else
        echo "  no ACS extended capability found on $bdf (or setpci failed)" >&2
    fi
done

echo
echo "Verify with: sudo ./acs.sh"
