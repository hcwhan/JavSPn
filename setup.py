import sys
import os
from typing import List, Tuple
from cx_Freeze import setup, Executable

# base="Win32GUI" should be used only for Windows GUI app
base = "Win32GUI" if sys.platform == "win32" else None

proj_root = os.path.abspath(os.path.dirname(__file__))

include_files: List[Tuple[str, str]] = [
        (f"{proj_root}/javspn/core/config.ini", 'config.ini'),
        (f"{proj_root}/data", 'data'),
        (f"{proj_root}/image", 'image')
]

setup(
    name='JavSPn',
    options = {'build_exe': {'include_files': include_files}}, 
    executables=[Executable("./javspn/__main__.py", target_name='JavSPn', base=base)]
)

