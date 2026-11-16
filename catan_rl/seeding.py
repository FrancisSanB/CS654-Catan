# Seeding the random generaters isnt enough to get the same game twice.
# Catanatron loops over sets and the set order changes every time python starts,
# unless PYTHONHASHSEED is fixed
import os
import sys


def ensure_hash_seed():
    # restarts the script with the hash seed set, call this first in main
    if os.environ.get("PYTHONHASHSEED") == "0":
        return
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable, *sys.argv])
