import subprocess
import sys
import argparse
import os
from pathlib import Path
import shutil
import platform

CURRENT_OS = str.lower(platform.system())

def run_command(command, env=None):
    print(f"\n[RUNNING] {' '.join(command)}")
    # 合并当前系统环境变量
    current_env = os.environ.copy()
    if env:
        current_env.update(env)
    subprocess.run(command, check=True, env=current_env)

def main():
    parser = argparse.ArgumentParser(description="arm NEON & SVE-128 Testing (Ubuntu) with QEMU")
    parser.add_argument("--test_mode", choices=["min", "max"], default="min")
    parser.add_argument("--sve_bits", type=int, choices=[128, 256, 512], default=128)
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    build_base = project_root / "build" / f"{CURRENT_OS}_arm"

    qemu_bin = shutil.which("qemu-aarch64-static") or shutil.which("qemu-aarch64")
    print(f"qemu bin path: {qemu_bin}")

    # 使用 aarch64 交叉工具链
    # (name, c_compiler, cxx_compiler, sub_dir)
    configs = [
        ("Clang-17", "clang-17", "clang++-17", "clang17"),
        ("GCC", "aarch64-linux-gnu-gcc-13", "aarch64-linux-gnu-g++-13", "gcc13"),
    ]
    if args.test_mode == "min":
        configs = configs[:1]

    build_options = [("Debug", "od")]
    if args.test_mode == "max":
        build_options += [("Release", "o2"), ("Release", "gl")]

    
    sve_bit_width = args.sve_bits; # SVE bit-width
    print(f"SVE bit width = {sve_bit_width}")

    for name, c_compiler, cxx_compiler, subdir in configs:
        for build_cfg, test_opt in build_options:
            current_build_dir = build_base / f"{subdir}_{build_cfg}_{test_opt}_sve_{sve_bit_width}"
            
            print(f"\n{'='*60}\nTarget: {name} | Config: {build_cfg} | Option: {test_opt}\n{'='*60}")

            # 1. 配置
            # name -> c_flags
            c_flags = {
                "Clang-17": "--target=aarch64-linux-gnu -march=armv8-a",
                "GCC": "-march=armv8-a"
            }

            cxx_flags = {
                "Clang-17": "--target=aarch64-linux-gnu -march=armv8-a",
                "GCC": "-march=armv8-a"
            }

            config_args = [
                "cmake", "-S", str(project_root), "-B", str(current_build_dir),
                "-G", "Ninja Multi-Config",
                f"-DCMAKE_C_COMPILER={c_compiler}",
                f"-DCMAKE_CXX_COMPILER={cxx_compiler}",
                f"-DCMAKE_C_FLAGS={c_flags[name]}",
                f"-DCMAKE_CXX_FLAGS={cxx_flags[name]}",
                "-DKSIMD_BUILD_TESTS=ON",
                f"-DKSIMD_TEST_OPTION={test_opt}",
                # 静态链接使得 QEMU 运行不需要额外的库搜索路径
                "-DCMAKE_EXE_LINKER_FLAGS=-static"
            ]
            if os.environ.get("GITHUB_ACTIONS") != "true":
                config_args.append(f"-DCMAKE_CROSSCOMPILING_EMULATOR={qemu_bin}")

            # 向cmake传递SVE宽度信息
            config_args.append(f"-DKSIMD_TEST_SVE_BITS={sve_bit_width}")

            run_command(config_args)

            # 2. 编译
            run_command(["cmake", "--build", str(current_build_dir), "--config", build_cfg])

            # 3. 测试
            width_str = str(sve_bit_width // 128)
            sve_test_env = {"QEMU_CPU": f"max,sve-max-vq={width_str}"}
            run_command([
                "ctest", "--output-on-failure", "--test-dir", str(current_build_dir), "-C", build_cfg
            ], env=sve_test_env)

    print("\n[SUCCESS] All ARM tests passed.")

if __name__ == "__main__":
    main()