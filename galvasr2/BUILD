py_binary(
    name = "galvasr_tokenize_words",
    srcs = ["galvasr_tokenize_words.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
    "//lingvo/core/ops:ops",
    "//lingvo/core:py_utils"
])

py_binary(
    name = "galvasr_latgen_faster",
    srcs = ["galvasr_latgen_faster.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
    "//lingvo/core/ops:ops",
    "//lingvo/core:py_utils",
    ],
    data = [
    "//third_party/kaldi:kaldi_pybind.so",
    ],
    imports = [
    "../third_party/kaldi"
    ]
)