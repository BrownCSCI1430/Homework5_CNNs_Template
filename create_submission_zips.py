import tempfile
import shutil
import os
from pathlib import Path
import sys
import glob
import re
from termcolor import colored


def print_err(message):
    print(colored("[ERROR] ", "red") + message)


def print_warn(message):
    print(colored("[WARNING] ", "magenta") + message)


def main():
    code_dir = Path("code")
    orig_dir = Path.cwd()

    # Try to find code directory
    if not code_dir.exists():
        # Might be running from within code/
        os.chdir(orig_dir.parent)
        if not code_dir.exists():
            os.chdir(orig_dir)

            # Try navigating to directory where script is
            os.chdir(os.path.dirname(sys.argv[0]))
            if not code_dir.exists():
                print_err("Failed finding code/ directory. Please make " +
                          "sure that you are running this script from within the " +
                          "directory in which create_submission_zip.py is " +
                          "located and that the code/ directory has not been " +
                          "renamed.")
                sys.exit(1)

    required_code_files = ["main.py", "hyperparameters.py", "models.py",
                           "preprocess.py", "tensorboard_utils.py"]

    # Check that required code files exist
    for code_file in required_code_files:
        if not (code_dir / code_file).exists():
            print_err("Could not find " + code_file + " within code/")
            sys.exit(1)

    # Create temp directory for copying needed files to
    temp_dir = Path(tempfile.mkdtemp())
    temp_code_dir = temp_dir / "code"
    os.mkdir(temp_code_dir)

    # Copy .py files to temp directory
    for code_file in glob.glob(str(code_dir / r"*.py")):
        shutil.copy(code_file, temp_code_dir)

    # Collect the reslog
    reslog_dir = Path("reslog")
    if not reslog_dir.exists():
        print_warn("Failed to find reslog directory.")
    else:
        temp_reslog_dir = temp_dir / "reslog"
        os.mkdir(temp_reslog_dir)
        shutil.copytree(reslog_dir, temp_reslog_dir, dirs_exist_ok=True)

    # Make code_reslog.zip
    shutil.make_archive(
        "code_reslog",
        format="zip",
        root_dir=temp_dir)
    shutil.rmtree(temp_dir)

    your_model_check_dir = Path("code") / "checkpoints" / "your_model"
    vgg_model_check_dir = Path("code") / "checkpoints" / "vgg_model"

    # Find best weight files
    def get_best_weights(name):
        check_dir = your_model_check_dir \
            if name == "your" \
            else vgg_model_check_dir

        if not check_dir.exists():
            print_warn("No checkpoint directory found for " + name + "_model. " +
                       "If you do have weights saved that you want to submit, " +
                       "please make sure they are in the checkpoints/ directory " +
                       "and that the structure of the directory is unchanged.")
        else:
            max_acc = 0
            max_acc_file = ""

            for root, _, files in os.walk(check_dir):
                for filename in files:
                    if filename.endswith(".weights.h5"):
                        file_acc = float(re.findall(
                            r"[+-]?\d+\.\d+", filename.split("acc")[-1])[0])

                        if file_acc > max_acc:
                            max_acc = file_acc
                            max_acc_file = Path(root) / filename

            if max_acc_file != "":
                # Create temp directory for copying weights to
                temp_dir = tempfile.mkdtemp()
                shutil.copy(max_acc_file, temp_dir)
                shutil.make_archive(
                    name + "_weights",
                    format="zip",
                    root_dir=temp_dir)
                shutil.rmtree(temp_dir)
            else:
                print_warn("No weight files found for " + name + "_model.")

    # Make your_weights.zip if weights exist
    get_best_weights("your")

    # Make vgg_weights.zip if weights exist
    get_best_weights("vgg")


if __name__ == "__main__":
    main()
