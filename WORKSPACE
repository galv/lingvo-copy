"""Workspace file for lingvo."""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load(
    "//lingvo:repo.bzl",
    "cc_tf_configure",
    "icu",
    "lingvo_protoc_deps",
    "lingvo_testonly_deps",
)

git_repository(
    name = "subpar",
    remote = "https://github.com/google/subpar",
    tag = "2.0.0",
)

git_repository(
    name = "openfst",
    commit = "22867ee7b483598a729abe125da2517584a5d010",  # 1.7.2 plus Bazel
    remote = "https://github.com/mjansche/openfst.git",
)

cc_tf_configure()

lingvo_testonly_deps()

lingvo_protoc_deps()

icu()

http_archive(
  name = "pybind11_bazel",
  strip_prefix = "pybind11_bazel-26973c0ff320cb4b39e45bc3e4297b82bc3a6c09",
  urls = ["https://github.com/pybind/pybind11_bazel/archive/26973c0ff320cb4b39e45bc3e4297b82bc3a6c09.zip"],
  sha256 = "a5666d950c3344a8b0d3892a88dc6b55c8e0c78764f9294e806d69213c03f19d"
)

http_archive(
  name = "pybind11",
  build_file = "@pybind11_bazel//:pybind11.BUILD",
  strip_prefix = "pybind11-2.5.0",
  urls = ["https://github.com/pybind/pybind11/archive/v2.5.0.tar.gz"],
  sha256 = "97504db65640570f32d3fdf701c25a340c8643037c3b69aec469c10c93dc8504"
)

load("@pybind11_bazel//:python_configure.bzl", "python_configure")
python_configure(name = "local_config_python")