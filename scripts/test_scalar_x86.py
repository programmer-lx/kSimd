import subprocess
import sys
import os
import argparse
from pathlib import Path
import shutil

def run_command(command, env=None):
    print(f"\n[RUNNING] {' '.join(command)}")
    subprocess.run(command, check=True, env=env)

def main():
    parser = argparse.ArgumentParser(description="X86 & Scalar Testing (Ubuntu) with QEMU")
    parser.add_argument("--test_mode", choices=["min", "max"], default="min", help="min: Debug/Clang, max: All configs/compilers")
    args = parser.parse_args()

    # 路径定义
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    build_base = project_root / "build" / "x86_scalar"

    qemu_bin = shutil.which("qemu-x86_64-static") or shutil.which("qemu-x86_64")
    print(f"qemu bin path: {qemu_bin}")

    # 编译器矩阵
    configs = [
        ("Clang-17", "clang-17", "clang++-17", "clang17"),
        ("GCC-13", "gcc-13", "g++-13", "gcc13")
    ]
    if args.test_mode == "min":
        configs = configs[:1]

    # 编译选项矩阵
    build_options = [("Debug", "od")]
    if args.test_mode == "max":
        build_options += [("Release", "o2"), ("Release", "gl")]

    for name, c_comp, cxx_comp, subdir in configs:
        for build_cfg, test_opt in build_options:
            current_build_dir = build_base / f"{subdir}_{build_cfg}_{test_opt}"
            
            print(f"\n{'='*60}\nTarget: {name} | Config: {build_cfg} | Option: {test_opt}\n{'='*60}")

            # 1. 配置
            config_args = [
                "cmake", "-S", str(project_root), "-B", str(current_build_dir),
                "-G", "Ninja Multi-Config",
                f"-DCMAKE_C_COMPILER={c_comp}",
                f"-DCMAKE_CXX_COMPILER={cxx_comp}",
                "-DKSIMD_BUILD_TESTS=ON",
                "-DKSIMD_CTEST_X86=ON",    # 开启 X86 变量
                f"-DKSIMD_TEST_OPTION={test_opt}",
                "-DCMAKE_EXE_LINKER_FLAGS=-static"
            ]
            if os.environ.get("GITHUB_ACTIONS") == "true":
                config_args.append(f"-DCMAKE_CROSSCOMPILING_EMULATOR={qemu_bin}")

            run_command(config_args)

            # 2. 编译
            run_command(["cmake", "--build", str(current_build_dir), "--config", build_cfg])

            # 3. 测试
            x86_env = {"QEMU_CPU": "max"}
            run_command([
                "ctest", "--output-on-failure", "--test-dir", str(current_build_dir), "-C", build_cfg
            ], env=x86_env)

    print("\n[SUCCESS] All X86 & Scalar tests passed.")

if __name__ == "__main__":
    main()