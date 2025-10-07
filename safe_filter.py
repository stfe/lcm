# Very simple allow/deny lists. Tune for your environment.
# Denies take precedence unless override flag is set.

DANGEROUS_PATTERNS = [
    "rm -rf /", "mkfs", "dd if=", "disks destroy", "format /",
    "shutdown -h now", "poweroff", "reboot", "kill -9 1",
    ">: /", ":> /", "truncate -s 0 /", "chown -R /", "chmod -R 000 /",
    "wipefs", "cryptsetup luksFormat", "mv /", "cp /dev/zero",
]

def is_safe(cmd: str) -> bool:
    c = cmd.strip().lower()
    for pat in DANGEROUS_PATTERNS:
        if pat in c:
            return False
    return True
