#!/bin/sh
#
# acsctl.sh - print the ACS Control register for every PCI device,
# grouped under each device's name. Run as root (sudo) or capabilities
# will show <access denied>.
#
# Any bit shown with '+' is enabled. For GPU P2P you want these all '-':
#   SrcValid, ReqRedir, CmpltRedir, UpstreamFwd
# Clear them with: sudo ./disable_acs_p2p.sh
#
# Fish shell equivalent (also installed as ~/.config/fish/functions/acsctl.fish):
#   function acsctl
#       sudo lspci -vvv | awk '...same awk program...'
#   end

lspci -vvv | awk '
    /^[0-9a-f][0-9a-f]:/ {
        dev = $0
        sub(/ \(prog-if.*/, "", dev)
        sub(/ \(rev [^)]*\)/, "", dev)
    }
    /ACSCtl:/ {
        sub(/^[ \t]+/, "")
        if ($0 ~ /\+/)
            print "\033[1;31m" dev "\n\t" $0 "\033[0m\n"
        else
            print dev "\n\t" $0 "\n"
    }
'

# One-liner:
# sudo lspci -vvv | awk '/^[0-9a-f][0-9a-f]:/{dev=$0; sub(/ \(prog-if.*/,"",dev); sub(/ \(rev [^)]*\)/,"",dev)} /ACSCtl:/{sub(/^[ \t]+/,""); if ($0 ~ /\+/) print "\033[1;31m" dev "\n\t" $0 "\033[0m\n"; else print dev "\n\t" $0 "\n"}'

