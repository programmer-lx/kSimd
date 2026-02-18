import subprocess
import sys
import argparse
import os
from pathlib import Path
import shutil

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
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    build_base = project_root / "build" / "arm_neon"

    qemu_bin = shutil.which("qemu-aarch64-static") or shutil.which("qemu-aarch64")
    print(f"qemu bin path: {qemu_bin}")

    # 使用 aarch64 交叉工具链
    c_compiler = "clang-17"
    cxx_compiler = "clang++-17"

    build_options = [("Debug", "od")]
    if args.test_mode == "max":
        build_options += [("Release", "o2")]

    for build_cfg, test_opt in build_options:
        current_build_dir = build_base / f"neon_{build_cfg}_{test_opt}"
        
        print(f"\n{'='*60}\nTarget: NEON (QEMU) | Config: {build_cfg} | Option: {test_opt}\n{'='*60}")

        # 1. 配置
        config_args = [
            "cmake", "-S", str(project_root), "-B", str(current_build_dir),
            "-G", "Ninja Multi-Config",
            f"-DCMAKE_C_COMPILER={c_compiler}",
            f"-DCMAKE_CXX_COMPILER={cxx_compiler}",
            "-DCMAKE_C_FLAGS=--target=aarch64-linux-gnu -march=armv9-a+sve",
            "-DCMAKE_CXX_FLAGS=--target=aarch64-linux-gnu -march=armv9-a+sve",
            "-DKSIMD_BUILD_TESTS=ON",
            "-DKSIMD_CTEST_NEON=ON",    # 开启 NEON 变量
            f"-DKSIMD_TEST_OPTION={test_opt}",
            # 静态链接使得 QEMU 运行不需要额外的库搜索路径
            "-DCMAKE_EXE_LINKER_FLAGS=-static"
        ]
        if os.environ.get("GITHUB_ACTIONS") == "true":
            config_args += f"-DCMAKE_CROSSCOMPILING_EMULATOR={qemu_bin}"

        run_command(config_args)

        # 2. 编译
        run_command(["cmake", "--build", str(current_build_dir), "--config", build_cfg])

        # 3. 测试
        sve_test_env = {"QEMU_CPU": "max,sve-max-vq=1"} # SVE-128
        run_command([
            "ctest", "--output-on-failure", "--test-dir", str(current_build_dir), "-C", build_cfg
        ], env=sve_test_env)

    print("\n[SUCCESS] All NEON tests passed via QEMU simulation.")

if __name__ == "__main__":
    main()