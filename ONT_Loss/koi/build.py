import os
from cffi import FFI

__dir__ = os.path.dirname(os.path.abspath(__file__))

lib = os.path.join(__dir__, "lib")
build = os.path.join(__dir__, "..", "build", "lib")
header = os.path.join(lib, "koi.h")

ffi = FFI()

ffi.set_source(
    'koi._runtime',
    '#include <koi.h>',
    include_dirs=[lib],
    library_dirs=[build],
    libraries=['koi']
)

with open(header, 'r') as f:
    ffi.cdef(f.read())


if __name__ == '__main__':
    ffi.compile()
