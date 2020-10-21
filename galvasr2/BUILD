py_binary(
    name = "galvasr_tokenize_words",
    srcs = ["galvasr_tokenize_words.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
    "//lingvo/core/ops:ops",
    "//lingvo/core:py_utils"
])

# This is unused and incomplete right now
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

# py_binary(
#     name = "galvasr_offline_wfst_decoder_ops",
#     srcs = ["galvasr_offline_wfst_decoder_ops.cc"],
#     python_version = "PY3",
#     srcs_version = "PY3",
#     deps = [
#     "//lingvo/core/ops:ops",
#     "//lingvo/core:py_utils",
#     ],
#     data = [
#     "//third_party/kaldi:kaldi",
#     ],
#     imports = [
#     "../third_party/kaldi"
#     ]
# )