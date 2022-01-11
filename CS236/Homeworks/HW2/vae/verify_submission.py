# Copyright (c) 2021 Rui Shu
import glob
import os
import subprocess

if not os.path.exists("hw2.zip"):
    raise FileNotFoundError("""Unable to find hw2.zip

    Why did this error occur?
    Common mistakes include:
    1. Running this code before you created your zip file
    2. Creating your zip file under a different name
    """.rstrip())

print("Unzipping hw2.zip to temporary directory test_zip")
subprocess.call("""
unzip hw2.zip -d test_zip
""", shell=True)

filepaths = ["test_zip/codebase/utils.py",
             "test_zip/codebase/models/vae.py",
             "test_zip/codebase/models/gmvae.py",
             "test_zip/codebase/models/ssvae.py",
             "test_zip/codebase/models/fsvae.py"]

filedirpaths = ["test_zip/codebase",
                "test_zip/codebase/models",
                "test_zip/codebase/utils.py",
                "test_zip/codebase/models/gmvae.py",
                "test_zip/codebase/models/ssvae.py",
                "test_zip/codebase/models/fsvae.py",
                "test_zip/codebase/models/vae.py"]

try:
    # Check that the files that matter exist
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        print("Checking for {} in {}".format(filename, filepath))
        if os.path.exists(filepath):
            print("...Found it!")
        else:
            raise FileNotFoundError(
                """
    Unable to find {0} in path {1}

    Why did this error occur?
    Common mistakes include:
    1. Zipping the files in a flat hierarchy instead of preserving the directories
    3. Zipping the files from the wrong root directory
    2. Forgetting into include {0} in your zip file (include all files,
       even if you did not complete the assignment for this file)
                """.format(filename, filepath).rstrip())

    # Check that no other files and directories exist
    paths = glob.glob("test_zip/**/*", recursive=True)
    for path in paths:
        if path not in filedirpaths:
            raise FileExistsError("""
     Zip file contains unpermitted file or directory: {0}

     Why did this error occur?
     Common mistakes include:
     1. Zipping your entire codebase
     2. Zipping your entire git repository... (why would you do this? ಠ_ಠ)
            """.format(path).rstrip())

finally:
    subprocess.call("""
    rm -r test_zip
    """, shell=True)
