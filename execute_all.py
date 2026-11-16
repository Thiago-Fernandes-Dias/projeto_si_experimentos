import os

SCRIPTS = [
    "st.py",
    "hocquet.py",
    "self_detector.py",
    "scaled_manhattan.py",
]

cmds = [
    f"python {filename}"
    for filename in SCRIPTS
    if os.path.isfile(os.path.join("./", filename))
]
os.system(" && ".join(cmds))
