"""
https://github.com/openslide/openslide-python/issues/115
https://github.com/openslide/openslide-python/issues/115#issuecomment-906888753
"""
import os

if os.name == 'nt':
    import ctypes.util
    _dll_path = ctypes.util.find_library("libopenslide-0.dll")
    if _dll_path is not None:
        _dll_directory, _ = os.path.split(os.path.abspath(_dll_path))
        if hasattr(os, 'add_dll_directory'):
            # Python >= 3.8
            with os.add_dll_directory(_dll_directory):
                import openslide
        else:
            # Python < 3.8
            _orig_path = os.environ.get('PATH', '')
            os.environ['PATH'] = _orig_path + ';' + _dll_path
            import openslide
            os.environ['PATH'] = _orig_path
else:
    import openslide
