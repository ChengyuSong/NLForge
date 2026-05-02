# Build environment for C/C++ projects with LLVM 18 toolchain
# Optimized for static analysis with LTO and LLVM IR generation

FROM ubuntu:24.04

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install LLVM 18 toolchain and build essentials
RUN apt-get update && apt-get install -y \
    # LLVM 18 toolchain
    clang-18 \
    clang++-18 \
    llvm-18 \
    llvm-18-dev \
    llvm-18-tools \
    lld-18 \
    libc++-18-dev \
    libc++abi-18-dev \
    # Build systems
    cmake \
    ninja-build \
    make \
    autoconf \
    automake \
    libtool \
    pkg-config \
    meson \
    scons \
    # Version control
    git \
    # Essential build tools
    build-essential \
    flex \
    bison \
    m4 \
    patch \
    python3 \
    python3-pip \
    python3-venv \
    perl \
    texinfo \
    gettext \
    autopoint \
    nasm \
    # Debugging
    gdb \
    # Utilities
    wget \
    curl \
    file \
    vim \
    bear \
    ccache \
    && rm -rf /var/lib/apt/lists/*

# Create ccache symlinks for clang-18 (Ubuntu's ccache package handles gcc/g++/cc/c++)
RUN ln -sf /usr/bin/ccache /usr/lib/ccache/clang-18 && \
    ln -sf /usr/bin/ccache /usr/lib/ccache/clang++-18

# Prepend ccache to PATH so compiler calls are intercepted
ENV PATH="/usr/lib/ccache:${PATH}"

# Install Bazelisk (auto-downloads correct Bazel version per project)
RUN wget -qO /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 && \
    chmod +x /usr/local/bin/bazel

# Set clang-18 as default compiler
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-18 100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-18 100 && \
    update-alternatives --install /usr/bin/llvm-link llvm-link /usr/bin/llvm-link-18 100 && \
    update-alternatives --install /usr/bin/llvm-ar llvm-ar /usr/bin/llvm-ar-18 100 && \
    update-alternatives --install /usr/bin/llvm-nm llvm-nm /usr/bin/llvm-nm-18 100 && \
    update-alternatives --install /usr/bin/opt opt /usr/bin/opt-18 100

# Default compiler environment variables
ENV CC=clang-18
ENV CXX=clang++-18
ENV AR=llvm-ar-18
ENV NM=llvm-nm-18
ENV RANLIB=llvm-ranlib-18
ENV LD=ld.lld-18

# Allow git to work with any repo ownership (mounted volumes may differ)
RUN git config --global --add safe.directory /workspace/src

# Working directory for builds
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
