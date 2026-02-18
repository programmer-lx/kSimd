import subprocess
import sys
import os
import argparse
from pathlib import Path
import shutil
import platform

CURRENT_OS = str.lower(platform.system())
SDE_NAME = str("")
if CURRENT_OS == "windows":
    SDE_NAME = "sde"
elif CURRENT_OS == "linux":
    SDE_NAME = "sde64"
else:
    print("Error: unknown os.")
    sys.exit(1)

def run_command(command, env=None):
    command = [str(c) for c in command]
    print(f"\n[RUNNING] {' '.join(command)}")
    
    current_env = os.environ.copy()
    if env:
        current_env.update(env)
        
    subprocess.run(command, check=True, env=current_env)

def main():
    parser = argparse.ArgumentParser(description="X86 Testing with Intel SDE (Sapphire Rapids)")
    parser.add_argument("--test_mode", choices=["min", "max"], default="min", help="min: Debug/Clang, max: All configs/compilers")
    args = parser.parse_args()

    # 路径定义
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    build_base = project_root / "build" / f"{CURRENT_OS}_x86"

    # 查找 SDE 可执行文件
    sde_bin = shutil.which(str(SDE_NAME))

    if not sde_bin:
        sde_root_env = os.environ.get("SDE_ROOT")
        if sde_root_env:
            # 拼接绝对路径并检查是否存在
            potential_sde = Path(sde_root_env) / (SDE_NAME + (".exe" if CURRENT_OS == "windows" else ""))
            if potential_sde.exists():
                sde_bin = str(potential_sde.resolve())

    if not sde_bin:
        print("Error: 'sde or sde64' not found in PATH. Please check SDE installation.")
        sys.exit(1)
    sde_bin = os.path.abspath(sde_bin)
    print(f"Using SDE found at: {sde_bin}")

    sde_root = os.path.dirname(sde_bin)
    os.environ["SDE_ROOT"] = sde_root
    print(f"Force setting SDE_ROOT to: {sde_root}")

    # 编译器矩阵
    config = []

    if CURRENT_OS == "windows":
        configs = [
            ("Clang-17", "clang", "clang++", "clang17"),
            ("MSVC", "cl", "cl", "msvc")
        ]
    
    if CURRENT_OS == "linux":
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

            # 1. configure
            config_args = [
                "cmake", "-S", project_root, "-B", current_build_dir,
                "-G", "Ninja Multi-Config",
                f"-DCMAKE_C_COMPILER={c_comp}",
                f"-DCMAKE_CXX_COMPILER={cxx_comp}",
                "-DKSIMD_BUILD_TESTS=ON",
                f"-DKSIMD_TEST_OPTION={test_opt}",
                "-DCMAKE_EXE_LINKER_FLAGS=-static",
                f"-DCMAKE_CROSSCOMPILING_EMULATOR={sde_bin};-spr;-verbose;10;-pin_home;{sde_root};--"
            ]

            run_command(config_args)

            # 2. 编译
            run_command(["cmake", "--build", current_build_dir, "--config", build_cfg])

            # 3. 测试
            ctest_bin = shutil.which("ctest")
            if not ctest_bin:
                print("Error: 'ctest' not found in PATH.")
                sys.exit(1)

            run_command([
                ctest_bin, "--output-on-failure", "--test-dir", current_build_dir, "-C", build_cfg
            ])

    print("\n[SUCCESS] All X86 tests passed.")

if __name__ == "__main__":
    main()