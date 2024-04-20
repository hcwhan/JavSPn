import os
import sys
import random
import string
import shutil
from glob import glob
import tempfile
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.config import cfg


def test_javsp_exe():
    cwd = os.getcwd()
    dist_dir = os.path.normpath(os.path.join(os.path.dirname(__file__) + '/../dist'))
    os.chdir(dist_dir)

    tmp_dir = tempfile.mkdtemp()
    print(f"Tempdir is {tmp_dir}")

    FILE = f"300MAAN-642.RIP.f4v"
    size = cfg.File.ignore_video_file_less_than
    size_MiB = int((size / (1 << 20)) + 1)

    try:
        os.system(f"dd if=/dev/zero of={FILE} bs=1MiB count={size_MiB}")

        exit_code = os.system(f"./JavSP --no-update --auto-exit --input . --output {tmp_dir}")
        assert exit_code == 0, f"Non-zero exit code: {exit_code}"
        # Check generated files
        files = glob(tmp_dir + '/**/*.*', recursive=True)
        assert all('横宮七海' in i for i in files), "Actress name not found"
        assert any(i.endswith('fanart.jpg') for i in files), "fanart not found"
        assert any(i.endswith('poster.jpg') for i in files), "poster not found"
        assert any(i.endswith('.f4v') for i in files), "video file not found"
        assert any(i.endswith('.nfo') for i in files), "nfo file not found"
    finally:
        # if os.path.exists(tmp_folder):
        #     shutil.rmtree(tmp_dir)
        os.chdir(cwd)

