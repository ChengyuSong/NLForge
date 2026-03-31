"""Build action tools for the build agent (CMake, Autotools, and arbitrary build systems)."""

import glob as globmod
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ..models import AssemblyCheckResult
from .assembly_utils import check_assembly
from .constants import (
    DOCKER_CCACHE_DIR,
    DOCKER_WORKSPACE_BUILD,
    DOCKER_WORKSPACE_SRC,
    TIMEOUT_BUILD,
    TIMEOUT_CONFIGURE,
    TIMEOUT_INSTALL,
    TIMEOUT_LONG_BUILD,
    TIMEOUT_RUN_COMMAND,
)


def _ccache_docker_args(ccache_dir: Path | None) -> list[str]:
    """Return docker run args to mount and configure ccache."""
    if ccache_dir is None:
        return []
    return [
        "-v", f"{ccache_dir}:{DOCKER_CCACHE_DIR}",
        "-e", f"CCACHE_DIR={DOCKER_CCACHE_DIR}",
    ]


class CMakeActions:
    """CMake configure and build actions for the agent."""

    def __init__(
        self,
        project_path: Path,
        build_dir: Path | None = None,
        container_image: str = "llm-summary-builder:latest",
        unavoidable_asm_path: Path | str | None = None,
        verbose: bool = False,
        ccache_dir: Path | None = None,
        source_subdir: str | None = None,
    ):
        self.project_path = Path(project_path).resolve()
        self.build_dir = Path(build_dir) if build_dir else self.project_path / "build"
        self.container_image = container_image
        self.unavoidable_asm_path = Path(unavoidable_asm_path) if unavoidable_asm_path else None
        self.verbose = verbose
        self.ccache_dir = ccache_dir
        self.source_subdir = source_subdir

    def cmake_configure(self, cmake_flags: list[str]) -> dict[str, Any]:
        """
        Run CMake configuration step in Docker.

        Args:
            cmake_flags: List of CMake flags (e.g., ["-DCMAKE_BUILD_TYPE=Release"])

        Returns:
            Dict with 'success' (bool), 'output' (str), 'error' (str)
        """
        try:
            self.build_dir.mkdir(parents=True, exist_ok=True)

            # Run as host user to avoid root-owned files
            uid = os.getuid()
            gid = os.getgid()

            # Build docker command
            docker_cmd = [
                "docker", "run", "--rm",
                "-u", f"{uid}:{gid}",
                *_ccache_docker_args(self.ccache_dir),
                "-v", f"{self.project_path}:{DOCKER_WORKSPACE_SRC}",
                "-v", f"{self.build_dir}:{DOCKER_WORKSPACE_BUILD}",
                "-w", DOCKER_WORKSPACE_BUILD,
                self.container_image,
                "bash", "-c",
            ]

            # Quote flags that contain spaces
            quoted_flags = []
            for flag in cmake_flags:
                if ' ' in flag:
                    quoted_flags.append(f"'{flag}'")
                else:
                    quoted_flags.append(flag)

            source_dir = (
                f"{DOCKER_WORKSPACE_SRC}/{self.source_subdir}"
                if self.source_subdir
                else DOCKER_WORKSPACE_SRC
            )
            cmake_cmd = f"cmake -G Ninja {' '.join(quoted_flags)} {source_dir}"

            if self.verbose:
                print(f"[cmake_configure] Running: {cmake_cmd}")

            docker_cmd.append(cmake_cmd)

            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_CONFIGURE,
            )

            output = result.stdout + result.stderr

            if result.returncode == 0:
                return {
                    "success": True,
                    "output": output,
                    "error": "",
                }
            else:
                return {
                    "success": False,
                    "output": output,
                    "error": f"CMake configure failed with exit code {result.returncode}",
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": f"CMake configure timed out after {TIMEOUT_CONFIGURE} seconds",
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": f"CMake configure failed: {str(e)}",
            }

    def cmake_build(self) -> dict[str, Any]:
        """
        Run ninja build in Docker.

        Returns:
            Dict with 'success' (bool), 'output' (str), 'error' (str)
        """
        try:
            if not self.build_dir.exists():
                return {
                    "success": False,
                    "output": "",
                    "error": "Build directory does not exist. Run cmake_configure first.",
                }

            # Check if CMakeCache exists (indicates successful configure)
            if not (self.build_dir / "CMakeCache.txt").exists():
                return {
                    "success": False,
                    "output": "",
                    "error": "CMakeCache.txt not found. Run cmake_configure successfully first.",
                }

            # Run as host user
            uid = os.getuid()
            gid = os.getgid()

            # Try parallel build first
            docker_cmd = [
                "docker", "run", "--rm",
                "-u", f"{uid}:{gid}",
                *_ccache_docker_args(self.ccache_dir),
                "-v", f"{self.project_path}:{DOCKER_WORKSPACE_SRC}",
                "-v", f"{self.build_dir}:{DOCKER_WORKSPACE_BUILD}",
                "-w", DOCKER_WORKSPACE_BUILD,
                self.container_image,
                "bash", "-c",
                "ninja -j$(nproc)",
            ]

            if self.verbose:
                print("[cmake_build] Running: ninja -j$(nproc)")

            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_BUILD,
            )

            output = result.stdout + result.stderr

            if result.returncode == 0:
                # Build succeeded - run assembly check
                asm_result = self._check_assembly()
                return {
                    "success": True,
                    "output": output,
                    "error": "",
                    "assembly_check": asm_result.to_dict() if asm_result else None,
                }
            else:
                # Parallel build failed - retry with -j1 for clearer error output
                if self.verbose:
                    print(
                        "[cmake_build] Parallel build failed,"
                        " retrying with ninja -j1 for clearer errors"
                    )

                docker_cmd_j1 = [
                    "docker", "run", "--rm",
                    "-u", f"{uid}:{gid}",
                    *_ccache_docker_args(self.ccache_dir),
                    "-v", f"{self.project_path}:{DOCKER_WORKSPACE_SRC}",
                    "-v", f"{self.build_dir}:{DOCKER_WORKSPACE_BUILD}",
                    "-w", DOCKER_WORKSPACE_BUILD,
                    self.container_image,
                    "bash", "-c",
                    "ninja -j1",
                ]

                result_j1 = subprocess.run(
                    docker_cmd_j1,
                    capture_output=True,
                    text=True,
                    timeout=TIMEOUT_BUILD,
                )

                output_j1 = result_j1.stdout + result_j1.stderr

                return {
                    "success": False,
                    "output": output_j1,  # Use j1 output - clearer errors
                    "error": f"Build failed with exit code {result_j1.returncode}",
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": f"Build timed out after {TIMEOUT_BUILD} seconds",
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": f"Build failed: {str(e)}",
            }

    def _check_assembly(self) -> "AssemblyCheckResult | None":
        """Run assembly verification after successful build."""
        return check_assembly(
            compile_commands_path=self.build_dir / "compile_commands.json",
            build_dir=self.build_dir,
            project_path=self.project_path,
            unavoidable_asm_path=self.unavoidable_asm_path,
            verbose=self.verbose,
            log_prefix="[cmake_build]",
        )


class AutotoolsActions:
    """Autotools configure and build actions for the agent."""

    def __init__(
        self,
        project_path: Path,
        build_dir: Path | None = None,
        container_image: str = "llm-summary-builder:latest",
        unavoidable_asm_path: Path | str | None = None,
        verbose: bool = False,
        ccache_dir: Path | None = None,
    ):
        self.project_path = Path(project_path).resolve()
        self.build_dir = Path(build_dir) if build_dir else self.project_path / "build"
        self.container_image = container_image
        self.unavoidable_asm_path = Path(unavoidable_asm_path) if unavoidable_asm_path else None
        self.verbose = verbose
        self.ccache_dir = ccache_dir

    def _get_default_env_flags(self) -> dict[str, str]:
        """Get default environment variables for autotools builds."""
        return {
            "CC": "clang-18",
            "CXX": "clang++-18",
            "CFLAGS": "-g -flto=full -save-temps=obj",
            "CXXFLAGS": "-g -flto=full -save-temps=obj",
            "LDFLAGS": "-flto=full -fuse-ld=lld",
            "LD": "ld.lld-18",
            "AR": "llvm-ar-18",
            "NM": "llvm-nm-18",
            "RANLIB": "llvm-ranlib-18",
        }

    def bootstrap(self, script_path: str = "bootstrap") -> dict[str, Any]:
        """
        Run a bootstrap script to prepare the build system.

        Args:
            script_path: Path to the bootstrap script relative to project root
                        (e.g., "bootstrap", "autogen.sh", "scripts/bootstrap.sh")

        Returns:
            Dict with 'success' (bool), 'output' (str), 'error' (str)
        """
        try:
            # Validate script path is within project directory
            script_full_path = (self.project_path / script_path).resolve()
            if not script_full_path.is_relative_to(self.project_path):
                return {
                    "success": False,
                    "output": "",
                    "error": f"Script path {script_path} is outside project directory",
                }

            if not script_full_path.exists():
                return {
                    "success": False,
                    "output": "",
                    "error": f"Bootstrap script not found: {script_path}",
                }

            # Get relative path from project root for docker execution
            script_rel = script_full_path.relative_to(self.project_path)

            # Run as host user to avoid root-owned files
            uid = os.getuid()
            gid = os.getgid()

            docker_cmd = [
                "docker", "run", "--rm",
                "-u", f"{uid}:{gid}",
                *_ccache_docker_args(self.ccache_dir),
                "-v", f"{self.project_path}:/workspace/src",
                "-w", DOCKER_WORKSPACE_SRC,
                self.container_image,
                "bash", "-c",
                f"chmod +x {script_rel} && ./{script_rel}",
            ]

            if self.verbose:
                print(f"[bootstrap] Running: {script_rel}")

            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_CONFIGURE,  # 5 minute timeout
            )

            output = result.stdout + result.stderr

            if result.returncode == 0:
                return {
                    "success": True,
                    "output": output,
                    "error": "",
                }
            else:
                return {
                    "success": False,
                    "output": output,
                    "error": f"bootstrap script failed with exit code {result.returncode}",
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": "bootstrap script timed out after 5 minutes",
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": f"bootstrap script failed: {str(e)}",
            }

    def autoreconf(self) -> dict[str, Any]:
        """
        Run autoreconf -fi to regenerate configure script.

        Returns:
            Dict with 'success' (bool), 'output' (str), 'error' (str)
        """
        try:
            # Run as host user to avoid root-owned files
            uid = os.getuid()
            gid = os.getgid()

            docker_cmd = [
                "docker", "run", "--rm",
                "-u", f"{uid}:{gid}",
                *_ccache_docker_args(self.ccache_dir),
                "-v", f"{self.project_path}:/workspace/src",
                "-w", DOCKER_WORKSPACE_SRC,
                self.container_image,
                "bash", "-c",
                "autoreconf -fi",
            ]

            if self.verbose:
                print("[autoreconf] Running: autoreconf -fi")

            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_CONFIGURE,  # 5 minute timeout
            )

            output = result.stdout + result.stderr

            if result.returncode == 0:
                return {
                    "success": True,
                    "output": output,
                    "error": "",
                }
            else:
                return {
                    "success": False,
                    "output": output,
                    "error": f"autoreconf failed with exit code {result.returncode}",
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": "autoreconf timed out after 5 minutes",
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": f"autoreconf failed: {str(e)}",
            }

    def run_configure(
        self,
        configure_flags: list[str],
        use_build_dir: bool = True,
    ) -> dict[str, Any]:
        """
        Run ./configure with flags.

        Args:
            configure_flags: List of configure flags (e.g., ["--disable-shared", "--enable-static"])
            use_build_dir: If True, use out-of-source build in /workspace/build

        Returns:
            Dict with 'success' (bool), 'output' (str), 'error' (str)
        """
        try:
            if use_build_dir:
                self.build_dir.mkdir(parents=True, exist_ok=True)

            # Run as host user
            uid = os.getuid()
            gid = os.getgid()

            # Build docker command
            docker_cmd = [
                "docker", "run", "--rm",
                "-u", f"{uid}:{gid}",
                *_ccache_docker_args(self.ccache_dir),
                "-v", f"{self.project_path}:/workspace/src",
            ]

            # Add build dir mount if using out-of-source build
            if use_build_dir:
                docker_cmd.extend([
                    "-v", f"{self.build_dir}:/workspace/build",
                    "-w", DOCKER_WORKSPACE_BUILD,
                ])
            else:
                docker_cmd.extend(["-w", DOCKER_WORKSPACE_SRC])

            docker_cmd.extend([self.container_image, "bash", "-c"])

            # Build environment variables
            env_flags = self._get_default_env_flags()
            env_str = " ".join(f'{k}="{v}"' for k, v in env_flags.items())

            # Quote flags that contain spaces
            quoted_flags = []
            for flag in configure_flags:
                if ' ' in flag:
                    quoted_flags.append(f"'{flag}'")
                else:
                    quoted_flags.append(flag)

            # Determine configure path
            if use_build_dir:
                configure_path = "/workspace/src/configure"
            else:
                configure_path = "./configure"

            configure_cmd = f"{env_str} {configure_path} {' '.join(quoted_flags)}"

            if self.verbose:
                print(f"[run_configure] Running: {configure_cmd}")

            docker_cmd.append(configure_cmd)

            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_BUILD,  # 10 minute timeout for configure
            )

            output = result.stdout + result.stderr

            if result.returncode == 0:
                return {
                    "success": True,
                    "output": output,
                    "error": "",
                    "use_build_dir": use_build_dir,
                }
            else:
                return {
                    "success": False,
                    "output": output,
                    "error": f"configure failed with exit code {result.returncode}",
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": "configure timed out after 10 minutes",
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": f"configure failed: {str(e)}",
            }

    def make_build(
        self,
        make_target: str = "",
        use_build_dir: bool = True,
    ) -> dict[str, Any]:
        """
        Run bear -- make to build and capture compile commands.

        Args:
            make_target: Optional make target (e.g., "all", empty for default)
            use_build_dir: If True, build in /workspace/build

        Returns:
            Dict with 'success' (bool), 'output' (str), 'error' (str), 'assembly_check' (optional)
        """
        try:
            # Determine working directory
            if use_build_dir:
                work_dir = self.build_dir
            else:
                work_dir = self.project_path

            if not work_dir.exists():
                return {
                    "success": False,
                    "output": "",
                    "error": f"Work directory does not exist: {work_dir}. Run run_configure first.",
                }

            # Check if Makefile exists (indicates successful configure)
            if not (work_dir / "Makefile").exists():
                return {
                    "success": False,
                    "output": "",
                    "error": "Makefile not found. Run run_configure successfully first.",
                }

            # Run as host user
            uid = os.getuid()
            gid = os.getgid()

            docker_cmd = [
                "docker", "run", "--rm",
                "-u", f"{uid}:{gid}",
                *_ccache_docker_args(self.ccache_dir),
                "-v", f"{self.project_path}:/workspace/src",
            ]

            if use_build_dir:
                docker_cmd.extend([
                    "-v", f"{self.build_dir}:/workspace/build",
                    "-w", DOCKER_WORKSPACE_BUILD,
                ])
            else:
                docker_cmd.extend(["-w", DOCKER_WORKSPACE_SRC])

            docker_cmd.extend([self.container_image, "bash", "-c"])

            # Build make command with bear
            target_str = make_target if make_target else ""
            make_cmd = f"bear -- make -j$(nproc) {target_str}".strip()

            if self.verbose:
                print(f"[make_build] Running: {make_cmd}")

            docker_cmd.append(make_cmd)

            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_LONG_BUILD,  # 20 minute timeout for build
            )

            output = result.stdout + result.stderr

            if result.returncode == 0:
                # Try 'bear --append -- make all' to capture any additional
                # targets (tests, tools) not built by the default make target.
                # This is a no-op if 'all' == default target.
                if not target_str or target_str != "all":
                    append_cmd = docker_cmd[:-1]  # Remove the original command
                    append_cmd.append("bear --append -- make -j$(nproc) all")
                    try:
                        subprocess.run(
                            append_cmd,
                            capture_output=True,
                            text=True,
                            timeout=TIMEOUT_LONG_BUILD,
                        )
                        # Ignore failure — 'make all' may not exist
                    except (subprocess.TimeoutExpired, Exception):
                        pass

                # Build succeeded - run assembly check
                asm_result = self._check_assembly(use_build_dir)
                return {
                    "success": True,
                    "output": output,
                    "error": "",
                    "assembly_check": asm_result.to_dict() if asm_result else None,
                }
            else:
                # Parallel build failed - retry with -j1 for clearer error output
                if self.verbose:
                    print(
                        "[make_build] Parallel build failed,"
                        " retrying with make -j1 for clearer errors"
                    )

                docker_cmd_j1 = docker_cmd[:-1]  # Remove the old command
                make_cmd_j1 = f"bear -- make -j1 {target_str}".strip()
                docker_cmd_j1.append(make_cmd_j1)

                result_j1 = subprocess.run(
                    docker_cmd_j1,
                    capture_output=True,
                    text=True,
                    timeout=TIMEOUT_LONG_BUILD,
                )

                output_j1 = result_j1.stdout + result_j1.stderr

                return {
                    "success": False,
                    "output": output_j1,  # Use j1 output - clearer errors
                    "error": f"Build failed with exit code {result_j1.returncode}",
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": "Build timed out after 20 minutes",
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": f"Build failed: {str(e)}",
            }

    def make_clean(
        self,
        use_build_dir: bool = True,
    ) -> dict[str, Any]:
        """
        Run make clean to remove compiled objects.

        Args:
            use_build_dir: If True, clean in /workspace/build

        Returns:
            Dict with 'success' (bool), 'output' (str), 'error' (str)
        """
        return self._run_make_target("clean", use_build_dir)

    def make_distclean(
        self,
        use_build_dir: bool = True,
    ) -> dict[str, Any]:
        """
        Run make distclean to remove all generated files including Makefile.

        Args:
            use_build_dir: If True, clean in /workspace/build

        Returns:
            Dict with 'success' (bool), 'output' (str), 'error' (str)
        """
        return self._run_make_target("distclean", use_build_dir)

    def _run_make_target(
        self,
        target: str,
        use_build_dir: bool = True,
    ) -> dict[str, Any]:
        """
        Run a make target (helper for clean/distclean).

        Args:
            target: Make target to run
            use_build_dir: If True, run in /workspace/build

        Returns:
            Dict with 'success' (bool), 'output' (str), 'error' (str)
        """
        try:
            # Determine working directory
            if use_build_dir:
                work_dir = self.build_dir
            else:
                work_dir = self.project_path

            if not work_dir.exists():
                return {
                    "success": False,
                    "output": "",
                    "error": f"Work directory does not exist: {work_dir}",
                }

            # Check if Makefile exists
            if not (work_dir / "Makefile").exists():
                return {
                    "success": False,
                    "output": "",
                    "error": "Makefile not found. Nothing to clean.",
                }

            # Run as host user
            uid = os.getuid()
            gid = os.getgid()

            docker_cmd = [
                "docker", "run", "--rm",
                "-u", f"{uid}:{gid}",
                *_ccache_docker_args(self.ccache_dir),
                "-v", f"{self.project_path}:/workspace/src",
            ]

            if use_build_dir:
                docker_cmd.extend([
                    "-v", f"{self.build_dir}:/workspace/build",
                    "-w", DOCKER_WORKSPACE_BUILD,
                ])
            else:
                docker_cmd.extend(["-w", DOCKER_WORKSPACE_SRC])

            docker_cmd.extend([self.container_image, "bash", "-c"])

            make_cmd = f"make {target}"

            if self.verbose:
                print(f"[make_{target}] Running: {make_cmd}")

            docker_cmd.append(make_cmd)

            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_CONFIGURE,  # 5 minute timeout
            )

            output = result.stdout + result.stderr

            if result.returncode == 0:
                return {
                    "success": True,
                    "output": output,
                    "error": "",
                }
            else:
                return {
                    "success": False,
                    "output": output,
                    "error": f"make {target} failed with exit code {result.returncode}",
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": f"make {target} timed out after 5 minutes",
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": f"make {target} failed: {str(e)}",
            }

    def _check_assembly(self, use_build_dir: bool = True) -> "AssemblyCheckResult | None":
        """Run assembly verification after successful build."""
        if use_build_dir:
            compile_commands_path = self.build_dir / "compile_commands.json"
            build_dir = self.build_dir
        else:
            compile_commands_path = self.project_path / "compile_commands.json"
            build_dir = self.project_path

        return check_assembly(
            compile_commands_path=compile_commands_path,
            build_dir=build_dir,
            project_path=self.project_path,
            unavoidable_asm_path=self.unavoidable_asm_path,
            verbose=self.verbose,
            log_prefix="[make_build]",
        )


# Package name validation pattern (prevent injection)
_PACKAGE_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9.+\-:]+$")


def install_packages(
    current_image: str,
    packages: list[str],
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Install system packages by building a derived Docker image.

    Creates a new Docker image extending current_image with the requested
    packages installed via apt-get. Returns the new image tag so subsequent
    docker run calls use the updated image.

    Args:
        current_image: The current Docker image to extend.
        packages: List of apt package names to install.
        verbose: Print debug info.

    Returns:
        Dict with 'success', 'output', 'error', 'new_image'.
    """
    if not packages:
        return {
            "success": False,
            "output": "",
            "error": "No packages specified.",
            "new_image": current_image,
        }

    # Validate package names
    invalid = [p for p in packages if not _PACKAGE_NAME_RE.match(p)]
    if invalid:
        return {
            "success": False,
            "output": "",
            "error": f"Invalid package name(s): {', '.join(invalid)}",
            "new_image": current_image,
        }

    # Build a deterministic tag from the sorted package list + base image
    pkg_key = ",".join(sorted(packages)) + ":" + current_image
    short_hash = hashlib.sha256(pkg_key.encode()).hexdigest()[:12]
    new_tag = f"llm-summary-builder:ext-{short_hash}"

    # Create temporary directory with Dockerfile
    dockerfile_content = (
        f"FROM {current_image}\n"
        f"USER root\n"
        f"RUN apt-get update && apt-get install -y {' '.join(packages)} "
        f"&& rm -rf /var/lib/apt/lists/*\n"
    )

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile_path = Path(tmpdir) / "Dockerfile"
            dockerfile_path.write_text(dockerfile_content)

            if verbose:
                print(f"[install_packages] Building derived image {new_tag}")
                print(f"[install_packages] Packages: {', '.join(packages)}")

            result = subprocess.run(
                ["docker", "build", "-t", new_tag, tmpdir],
                capture_output=True,
                text=True,
                timeout=TIMEOUT_INSTALL,
            )

            output = result.stdout + result.stderr

            if result.returncode == 0:
                if verbose:
                    print(f"[install_packages] Image {new_tag} built successfully")
                return {
                    "success": True,
                    "output": output,
                    "error": "",
                    "new_image": new_tag,
                }
            else:
                return {
                    "success": False,
                    "output": output,
                    "error": f"docker build failed with exit code {result.returncode}",
                    "new_image": current_image,
                }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "output": "",
            "error": f"Package installation timed out after {TIMEOUT_INSTALL} seconds",
            "new_image": current_image,
        }
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": f"Package installation failed: {str(e)}",
            "new_image": current_image,
        }


# Tool definitions for LLM (Anthropic tool use format)
# CMake tools
CMAKE_TOOL_DEFINITIONS = [
    {
        "name": "cmake_configure",
        "description": (
            "Run CMake configuration step. This generates build files but does not compile. "
            "Use this to test different CMake flags before building. The command runs in a Docker "
            "container with the project mounted at /workspace/src and build at /workspace/build."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "cmake_flags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "CMake flags like "
                        "['-DCMAKE_BUILD_TYPE=Release', "
                        "'-DBUILD_SHARED_LIBS=OFF']. "
                        "Always include: "
                        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON, "
                        "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON, "
                        "-DCMAKE_C_COMPILER=clang-18, "
                        "-DCMAKE_CXX_COMPILER=clang++-18, "
                        "-DBUILD_SHARED_LIBS=OFF, "
                        "-DCMAKE_C_FLAGS="
                        "'-g -flto=full -save-temps=obj', "
                        "-DCMAKE_CXX_FLAGS="
                        "'-g -flto=full -save-temps=obj -stdlib=libc++'"
                    ),
                }
            },
            "required": ["cmake_flags"],
        },
    },
    {
        "name": "cmake_build",
        "description": (
            "Run the build step (ninja) after cmake configure "
            "has succeeded. Only call this after a successful "
            "cmake_configure. The build runs in a Docker "
            "container. On success, returns an assembly_check "
            "result showing if any assembly code (.s files, "
            "inline asm) was compiled. Assembly is acceptable "
            "but you may try one round of flags to reduce it "
            "(e.g., -DDISABLE_ASM=ON, -DENABLE_SIMD=OFF)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
]

# Configure/Make tools
CONFIGURE_MAKE_TOOL_DEFINITIONS = [
    {
        "name": "bootstrap",
        "description": (
            "Run a bootstrap script to prepare the build system before configure. "
            "Many autotools projects provide a bootstrap script (bootstrap, autogen.sh, buildconf) "
            "that must be run before ./configure. Use this if the project has such a script. "
            "The script must be in the PROJECT directory "
            "(NOT 'build/'). Absolute paths are NOT allowed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "script_path": {
                    "type": "string",
                    "description": (
                        "Relative path to bootstrap script in project directory. "
                        "Examples: 'bootstrap', 'autogen.sh', 'buildconf', 'scripts/bootstrap.sh'. "
                        "Do NOT use 'build/' prefix. Default is 'bootstrap'."
                    ),
                    "default": "bootstrap",
                },
            },
        },
    },
    {
        "name": "autoreconf",
        "description": (
            "Run autoreconf -fi to regenerate the configure script from configure.ac. "
            "Use this when the project has configure.ac but no configure script, or when "
            "configure.ac has been modified. This is typically needed for projects cloned "
            "from version control. The command runs in a Docker container."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "run_configure",
        "description": (
            "Run ./configure with flags. This generates Makefile but does not compile. "
            "Environment variables CC, CXX, CFLAGS, CXXFLAGS, LD, AR, NM, RANLIB are "
            "automatically set to use clang-18 with LTO. The command runs in a Docker "
            "container with the project mounted at /workspace/src."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "configure_flags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Configure flags like ['--disable-shared', "
                        "'--enable-static', '--prefix=/usr']. "
                        "Common flags for minimizing assembly: --disable-asm, --disable-simd, "
                        "--disable-hardware-acceleration."
                    ),
                },
                "use_build_dir": {
                    "type": "boolean",
                    "description": (
                        "If true (default), use out-of-source build in /workspace/build. "
                        "If false, build in-source in /workspace/src. Out-of-source is cleaner "
                        "but some projects require in-source builds."
                    ),
                    "default": True,
                },
            },
            "required": ["configure_flags"],
        },
    },
    {
        "name": "make_build",
        "description": (
            "Run 'bear -- make' to build and capture compile commands. Only call this after "
            "a successful run_configure, or directly if only a Makefile exists. Bear wraps "
            "make to generate compile_commands.json. "
            "On success, returns an assembly_check result showing if any assembly code (.s files, "
            "inline asm) was compiled. Assembly is acceptable but you may try one round of "
            "configure flags to reduce it (e.g., --disable-asm, --disable-simd)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "make_target": {
                    "type": "string",
                    "description": (
                        "Optional make target (e.g., 'all', 'lib'). Leave empty for default target."
                    ),
                    "default": "",
                },
                "use_build_dir": {
                    "type": "boolean",
                    "description": (
                        "Must match the use_build_dir setting from run_configure. "
                        "Set to false if building directly with Makefile in source directory."
                    ),
                    "default": True,
                },
            },
        },
    },
    {
        "name": "make_clean",
        "description": (
            "Run 'make clean' to remove compiled object files. Use this before retrying a build "
            "with different flags, or to clean up after a failed build. Does not remove Makefile "
            "or configuration files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "use_build_dir": {
                    "type": "boolean",
                    "description": (
                        "Must match the use_build_dir setting from run_configure."
                    ),
                    "default": True,
                },
            },
        },
    },
    {
        "name": "make_distclean",
        "description": (
            "Run 'make distclean' to remove all generated files including Makefile and "
            "configuration cache. Use this before reconfiguring with completely different flags. "
            "After distclean, you must run run_configure again before building."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "use_build_dir": {
                    "type": "boolean",
                    "description": (
                        "Must match the use_build_dir setting from run_configure."
                    ),
                    "default": True,
                },
            },
        },
    },
]

# Shared finish tool for all build systems
FINISH_TOOL_DEFINITION = {
    "name": "finish",
    "description": (
        "Signal that the build task is complete. Call this when: "
        "(1) build succeeded with acceptable assembly results, "
        "(2) you've identified unresolvable blockers, or "
        "(3) you've exhausted all reasonable options. "
        "Do NOT call build tools after a successful build - call finish instead. "
        "If you installed packages with install_packages, report which ones are actual "
        "build dependencies in the 'dependencies' field. "
        "If you used test_build_script, include the validated script in build_script."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["success", "failure"],
                "description": (
                    "Final status: 'success' if build completed,"
                    " 'failure' if blocked or impossible"
                ),
            },
            "summary": {
                "type": "string",
                "description": "Brief summary of what was accomplished or why it failed",
            },
            "dependencies": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "List of apt package names that are actual build dependencies "
                    "(packages you installed via install_packages that the project needs). "
                    "Only include packages that were genuinely required for the build."
                ),
            },
            "build_script": {
                "type": "string",
                "description": (
                    "The validated build script content "
                    "(from a successful test_build_script "
                    "call). Include this for non-CMake/"
                    "non-Autotools builds so the script "
                    "can be reproduced."
                ),
            },
            "build_system": {
                "type": "string",
                "enum": ["cmake", "autotools", "meson", "bazel", "scons", "custom"],
                "description": (
                    "The build system used. 'cmake' and 'autotools' are auto-detected from "
                    "structured tools. For other build systems, report the type: 'meson', "
                    "'bazel', 'scons', or 'custom' if truly unknown."
                ),
            },
        },
        "required": ["status", "summary"],
    },
}

# Request more turns tool
REQUEST_MORE_TURNS_TOOL_DEFINITION = {
    "name": "request_more_turns",
    "description": (
        "Request additional turns when running low. Use this when you are making progress "
        "but need more turns to complete the build (e.g., complex project with many "
        "dependencies, multiple build iterations needed). Grants 10 extra turns per call."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": (
                    "Brief explanation of why more turns "
                    "are needed and what remains to be done."
                ),
            },
        },
        "required": ["reason"],
    },
}

# Install packages tool
INSTALL_PACKAGES_TOOL_DEFINITION = {
    "name": "install_packages",
    "description": (
        "Install system packages (apt) into the build environment. Use this when a build "
        "fails due to missing headers or libraries (e.g., 'fatal error: zlib.h: No such file "
        "or directory' → install zlib1g-dev). This builds a derived Docker image with the "
        "requested packages. Subsequent build commands will use the updated image. "
        "When calling finish, report which installed packages are actual dependencies."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "packages": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "List of apt package names to install (e.g., ['zlib1g-dev', 'libssl-dev']). "
                    "Use Debian/Ubuntu package names."
                ),
            },
        },
        "required": ["packages"],
    },
}


class RunCommandAction:
    """Run arbitrary shell commands in Docker for non-CMake/non-Autotools build systems."""

    def __init__(
        self,
        project_path: Path,
        build_dir: Path | None = None,
        container_image: str = "llm-summary-builder:latest",
        verbose: bool = False,
        ccache_dir: Path | None = None,
    ):
        self.project_path = Path(project_path).resolve()
        self.build_dir = Path(build_dir) if build_dir else self.project_path / "build"
        self.container_image = container_image
        self.verbose = verbose
        self.ccache_dir = ccache_dir

    def run_command(self, command: str, workdir: str = "src") -> dict[str, Any]:
        """
        Run an arbitrary shell command in Docker.

        Args:
            command: Shell command to run via bash -c
            workdir: "src" for /workspace/src, "build" for /workspace/build

        Returns:
            Dict with 'success' (bool), 'output' (str), 'error' (str)
        """
        try:
            self.build_dir.mkdir(parents=True, exist_ok=True)

            uid = os.getuid()
            gid = os.getgid()

            if workdir == "build":
                work_path = DOCKER_WORKSPACE_BUILD
            else:
                work_path = DOCKER_WORKSPACE_SRC

            docker_cmd = [
                "docker", "run", "--rm",
                "-u", f"{uid}:{gid}",
                *_ccache_docker_args(self.ccache_dir),
                "-v", f"{self.project_path}:{DOCKER_WORKSPACE_SRC}",
                "-v", f"{self.build_dir}:{DOCKER_WORKSPACE_BUILD}",
                "-w", work_path,
                self.container_image,
                "bash", "-c",
                command,
            ]

            if self.verbose:
                print(f"[run_command] Running in {workdir}: {command[:200]}")

            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_RUN_COMMAND,
            )

            output = result.stdout + result.stderr

            if result.returncode == 0:
                return {
                    "success": True,
                    "output": output,
                    "error": "",
                }
            else:
                return {
                    "success": False,
                    "output": output,
                    "error": f"Command failed with exit code {result.returncode}",
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": f"Command timed out after {TIMEOUT_RUN_COMMAND} seconds",
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": f"Command failed: {str(e)}",
            }


def test_build_script(
    script_content: str,
    project_path: Path,
    build_dir: Path,
    container_image: str = "llm-summary-builder:latest",
    unavoidable_asm_path: Path | None = None,
    verbose: bool = False,
    ccache_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Test a build script by running it from scratch in a clean temporary build directory.

    Verifies that the script produces compile_commands.json (required).
    On success, moves the entire build output to build_dir.

    Args:
        script_content: Shell script content to run
        project_path: Path to project source
        build_dir: Real build directory (artifacts copied here on success)
        container_image: Docker image to use
        unavoidable_asm_path: Path to unavoidable assembly findings
        verbose: Print debug info

    Returns:
        Dict with 'success', 'output', 'error', 'assembly_check', 'compile_commands_found'
    """
    project_path = Path(project_path).resolve()
    build_dir = Path(build_dir)
    temp_build_dir = None

    try:
        # Create a temp build directory for a clean test
        temp_build_dir = tempfile.mkdtemp(prefix="llm-summary-test-build-")

        if verbose:
            print(f"[test_build_script] Testing script in temp dir: {temp_build_dir}")

        uid = os.getuid()
        gid = os.getgid()

        docker_cmd = [
            "docker", "run", "--rm",
            "-u", f"{uid}:{gid}",
            *_ccache_docker_args(ccache_dir),
            "-v", f"{project_path}:{DOCKER_WORKSPACE_SRC}",
            "-v", f"{temp_build_dir}:{DOCKER_WORKSPACE_BUILD}",
            "-w", DOCKER_WORKSPACE_BUILD,
            container_image,
            "bash", "-c",
            script_content,
        ]

        if verbose:
            print(f"[test_build_script] Running script ({len(script_content)} chars)...")

        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_LONG_BUILD,  # 20 minutes for full build
        )

        output = result.stdout + result.stderr

        if result.returncode != 0:
            return {
                "success": False,
                "output": output,
                "error": f"Script failed with exit code {result.returncode}",
                "compile_commands_found": False,
            }

        # Check for compile_commands.json in temp build dir and project dir
        compile_commands_path = None
        for candidate in [
            Path(temp_build_dir) / "compile_commands.json",
            project_path / "compile_commands.json",
        ]:
            if candidate.exists():
                compile_commands_path = candidate
                break

        # Also search subdirectories of temp build dir
        if compile_commands_path is None:
            for found in globmod.glob(
                str(Path(temp_build_dir) / "**" / "compile_commands.json"),
                recursive=True,
            ):
                compile_commands_path = Path(found)
                break

        if compile_commands_path is None:
            return {
                "success": False,
                "output": output,
                "error": (
                    "compile_commands.json not generated. "
                    "The build script must produce compile_commands.json. "
                    "For CMake: use -DCMAKE_EXPORT_COMPILE_COMMANDS=ON. "
                    "For Make: wrap with 'bear -- make'. "
                    "For other build systems: use bear or compiledb."
                ),
                "compile_commands_found": False,
            }

        # Detect in-tree build (compile_commands.json in source, not build dir)
        in_tree_build = compile_commands_path.is_relative_to(project_path)

        if verbose:
            print(f"[test_build_script] Found compile_commands.json at {compile_commands_path}")
            if in_tree_build:
                print("[test_build_script] Detected in-tree build "
                      "(compile_commands.json in source dir)")

        # Run assembly check
        asm_result = check_assembly(
            compile_commands_path=compile_commands_path,
            build_dir=Path(temp_build_dir),
            project_path=project_path,
            unavoidable_asm_path=unavoidable_asm_path,
            verbose=verbose,
            log_prefix="[test_build_script]",
        )

        # Success — move all build content to the real build dir
        if build_dir.exists():
            shutil.rmtree(build_dir)
        shutil.move(str(temp_build_dir), str(build_dir))
        temp_build_dir = None  # prevent cleanup in finally block

        if verbose:
            print(f"[test_build_script] Moved build output to {build_dir}")

        return {
            "success": True,
            "output": output,
            "error": "",
            "compile_commands_found": True,
            "in_tree_build": in_tree_build,
            "assembly_check": asm_result.to_dict() if asm_result else None,
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "output": "",
            "error": f"Build script timed out after {TIMEOUT_LONG_BUILD} seconds",
            "compile_commands_found": False,
        }
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": f"test_build_script failed: {str(e)}",
            "compile_commands_found": False,
        }
    finally:
        # Clean up temp directory
        if temp_build_dir and os.path.exists(temp_build_dir):
            try:
                shutil.rmtree(temp_build_dir)
                if verbose:
                    print(f"[test_build_script] Cleaned up temp dir: {temp_build_dir}")
            except Exception:
                pass


def check_assembly_tool(
    build_dir: Path,
    project_path: Path,
    unavoidable_asm_path: Path | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Standalone assembly check tool for use after run_command builds.

    Args:
        build_dir: Build output directory
        project_path: Source project directory
        unavoidable_asm_path: Path to unavoidable assembly findings
        verbose: Print debug info

    Returns:
        Dict with assembly check results or error
    """
    # Search for compile_commands.json in build dir and project dir
    compile_commands_path = None
    for candidate in [
        Path(build_dir) / "compile_commands.json",
        Path(project_path) / "compile_commands.json",
    ]:
        if candidate.exists():
            compile_commands_path = candidate
            break

    if compile_commands_path is None:
        return {
            "success": False,
            "error": "compile_commands.json not found in build or project directory",
        }

    asm_result = check_assembly(
        compile_commands_path=compile_commands_path,
        build_dir=Path(build_dir),
        project_path=Path(project_path),
        unavoidable_asm_path=unavoidable_asm_path,
        verbose=verbose,
        log_prefix="[check_assembly]",
    )

    if asm_result is None:
        return {
            "success": True,
            "assembly_check": None,
            "message": "Assembly check could not be performed (no compile_commands.json or error)",
        }

    return {
        "success": True,
        "assembly_check": asm_result.to_dict(),
    }


# Tool definitions for arbitrary build systems
RUN_COMMAND_TOOL_DEFINITION = {
    "name": "run_command",
    "description": (
        "Run an arbitrary shell command in Docker for exploring and building projects "
        "that use Meson, Bazel, SCons, or custom build systems. The command runs via "
        "'bash -c' in the Docker container with the project at /workspace/src and build "
        "dir at /workspace/build. Use this for trial and error during exploration. "
        "After figuring out how to build, distill working commands into a script and "
        "call test_build_script to verify reproducibility."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": (
                    "Shell command to run. Examples: 'meson setup /workspace/build', "
                    "'ninja -C /workspace/build', 'bazel build //...', 'make -j$(nproc)'"
                ),
            },
            "workdir": {
                "type": "string",
                "enum": ["src", "build"],
                "description": (
                    "Working directory: 'src' for /workspace/src (project root), "
                    "'build' for /workspace/build. Default: 'src'"
                ),
                "default": "src",
            },
        },
        "required": ["command"],
    },
}

TEST_BUILD_SCRIPT_TOOL_DEFINITION = {
    "name": "test_build_script",
    "description": (
        "Test a build script by running it from scratch in a clean temporary build "
        "directory. Verifies that the script produces compile_commands.json (required). "
        "Also runs assembly check. Use this after you've figured out how to build with "
        "run_command to verify your script is reproducible before calling finish(). "
        "If it fails, adjust the script and retry."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "script": {
                "type": "string",
                "description": (
                    "Shell script content. Will be run via bash -c inside Docker with "
                    "project at /workspace/src and a clean build dir at /workspace/build. "
                    "Must generate compile_commands.json. For CMake use "
                    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON, for Make use 'bear -- make'."
                ),
            },
        },
        "required": ["script"],
    },
}

CHECK_ASSEMBLY_TOOL_DEFINITION = {
    "name": "check_assembly",
    "description": (
        "Check for assembly code in the build output. Use this after a successful "
        "run_command build to check if any assembly (.s files, inline asm) was compiled. "
        "Requires compile_commands.json to exist in the build or project directory. "
        "For CMake/Autotools builds, assembly check runs automatically after cmake_build "
        "or make_build — you don't need this tool for those."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
    },
}

