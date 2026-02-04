from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os

ASCEND_HOME = os.environ.get('ASCEND_HOME_PATH', '/usr/local/Ascend/ascend-toolkit/latest')
CUSTOM_OPP = f"{ASCEND_HOME}/opp/vendors/customize"

ext_modules = [
    Pybind11Extension(
        "relu_custom_npu",
        ["relu_custom_pybind.cpp"],
        include_dirs=[
            f"{ASCEND_HOME}/include",
            f"{CUSTOM_OPP}/op_api/include",
        ],
        library_dirs=[
            f"{ASCEND_HOME}/lib64",
            f"{CUSTOM_OPP}/op_api/lib",
        ],
        libraries=["ascendcl", "nnopbase", "acl_op_compiler", "cust_opapi"],
        extra_link_args=[
            f"-Wl,-rpath,{ASCEND_HOME}/lib64",
            f"-Wl,-rpath,{CUSTOM_OPP}/op_api/lib",
        ],
        cxx_std=17,
    ),
]

setup(
    name="relu_custom_npu",
    version="1.0.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
