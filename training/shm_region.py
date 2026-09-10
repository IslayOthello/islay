"""Private POSIX shared pages, unlinked immediately and inherited only through explicit fds."""
import ctypes
import mmap
import os
import secrets


class Region:
    def __init__(self, size):
        if os.name != "posix" or not 0 < size <= 128 * 1024 * 1024:
            raise ValueError("expected bounded POSIX shared memory")
        library = ctypes.CDLL(None, use_errno=True)
        library.shm_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint]
        library.shm_open.restype = ctypes.c_int
        library.shm_unlink.argtypes = [ctypes.c_char_p]
        library.shm_unlink.restype = ctypes.c_int
        name = ("/islay-" + secrets.token_hex(10)).encode()
        self.fd = library.shm_open(name, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        if self.fd < 0: raise OSError(ctypes.get_errno(), "shm_open failed")
        try:
            if library.shm_unlink(name): raise OSError(ctypes.get_errno(), "shm_unlink failed")
            os.set_inheritable(self.fd, False)
            os.ftruncate(self.fd, size)
            self.map = mmap.mmap(self.fd, size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        except BaseException:
            os.close(self.fd)
            raise

    def close(self):
        self.map.close()
        os.close(self.fd)

    def __enter__(self): return self
    def __exit__(self, *exception): self.close()
