// clang-format off

#include "kSimd/core/impl/base.hpp"

// arch headers
#if KSIMD_ARCH_X86_ANY
    #if defined(_MSC_VER)
        #include <intrin.h>
    #else
        #include <cpuid.h>
    #endif
#elif KSIMD_ARCH_ARM_ANY
    #if KSIMD_OS_LINUX
        #include <sys/auxv.h>
        #include <asm/hwcap.h>
    #elif KSIMD_OS_MACOS
        #include <sys/sysctl.h>
    #elif KSIMD_OS_WINDOWS
        // Windows on ARM
        #include <windows.h>
    #else
        #error unknown arm OS.
    #endif
#else
    #error Unknown arch
#endif

#include <cstddef>

namespace
{
    template<typename T, typename E>
    constexpr bool bit_is_open(T value, E bit) noexcept
    {
        static_assert(std::is_unsigned_v<T>, "T must be unsigned.");

        return ( value & (T(1) << static_cast<T>(bit)) ) != 0;
    }

    #if KSIMD_ARCH_X86_ANY
    enum class CpuFeatureIndex_EAX1_ECX0 : uint32_t
    {
        // see https://en.wikipedia.org/wiki/CPUID

        // ECX
        SSE3        = 0 , // EAX 1 ECX 0, ECX  0
        SSSE3       = 9 , // EAX 1 ECX 0, ECX  9
        FMA3        = 12, // EAX 1 ECX 0, ECX 12
        SSE4_1      = 19, // EAX 1 ECX 0, ECX 19
        SSE4_2      = 20, // EAX 1 ECX 0, ECX 20
        POPCNT      = 23, // EAX 1 ECX 0, ECX 23
        XSAVE       = 26, // EAX 1 ECX 0, ECX 26
        OS_XSAVE    = 27, // EAX 1 ECX 0, ECX 27
        AVX         = 28, // EAX 1 ECX 0, ECX 28
        F16C        = 29, // EAX 1 ECX 0, ECX 29

        // EDX
        FXSR        = 24, // EAX 1 ECX 0, EDX 24
        SSE         = 25, // EAX 1 ECX 0, EDX 25
        SSE2        = 26, // EAX 1 ECX 0, EDX 26
    };

    enum class CpuFeatureIndex_EAX7_ECX0 : uint32_t
    {
        AVX2        = 5 , // EAX 7 ECX 0, EBX  5
        AVX_512_F   = 16, // EAX 7 ECX 0, EBX 16
    };

    enum class CpuXSaveStateIndex : uint64_t
    {
        // see https://en.wikipedia.org/wiki/CPUID XSAVE State-components

        SSE                 = 1 , // XMM0-XMM15 and MXCSR
        AVX                 = 2 , // YMM0-YMM15
        AVX_512_K0_K7       = 5 , // opmask registers k0-k7
        AVX_512_LOW_256     = 6 , // ZMM0-ZMM15
        AVX_512_HIGH_256    = 7 , // ZMM16-ZMM31
    };

    // leaf: EAX, sub_leaf: ECX
    void cpuid(const uint32_t leaf, const uint32_t sub_leaf, uint32_t* abcd)
    {
        #if KSIMD_COMPILER_MSVC

        int regs[4];
        __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(sub_leaf));
        for (int i = 0; i < 4; ++i)
        {
            abcd[i] = static_cast<uint32_t>(regs[i]);
        }

        #elif KSIMD_COMPILER_GCC || KSIMD_COMPILER_CLANG

        uint32_t a;
        uint32_t b;
        uint32_t c;
        uint32_t d;
        __cpuid_count(leaf, sub_leaf, a, b, c, d);
        abcd[0] = a;
        abcd[1] = b;
        abcd[2] = c;
        abcd[3] = d;

        #else
        #error unknown compiler
        #endif
    }

    uint64_t xgetbv(uint32_t idx)
    {
        #if KSIMD_COMPILER_MSVC

        return _xgetbv(idx);

        #elif KSIMD_COMPILER_GCC || KSIMD_COMPILER_CLANG

        uint32_t eax, edx;
        __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(idx));
        return (static_cast<uint64_t>(edx) << 32) | eax;

        #else
        #error unknown compiler
        #endif
    }
    #endif // arch x86 any
}

namespace ksimd
{
    const CpuSupportInfo& get_cpu_support_info() noexcept
    {
        static CpuSupportInfo info = []()
        {
            CpuSupportInfo result{};

            #if KSIMD_ARCH_X86_ANY
            uint32_t abcd[4]; // eax, ebx, ecx, edx

            cpuid(0, 0, abcd);
            const uint32_t max_leaf = abcd[0];
            uint64_t xcr0 = 0;


            // ------------------ EAX 1 ECX 0 ------------------
            if (max_leaf >= 1)
            {
                // 查询 EAX 1, ECX 0
                cpuid(1, 0, abcd);
                const uint32_t ecx = abcd[2];
                const uint32_t edx = abcd[3];

                result.POPCNT = bit_is_open(ecx, CpuFeatureIndex_EAX1_ECX0::POPCNT);

                // ------------------------- FXSR -------------------------
                result.FXSR = bit_is_open(edx, CpuFeatureIndex_EAX1_ECX0::FXSR);

                // ------------------------- SSE family -------------------------
                result.SSE = result.FXSR && bit_is_open(edx, CpuFeatureIndex_EAX1_ECX0::SSE);
                result.SSE2 = result.SSE && bit_is_open(edx, CpuFeatureIndex_EAX1_ECX0::SSE2);
                result.SSE3 = result.SSE2 && bit_is_open(ecx, CpuFeatureIndex_EAX1_ECX0::SSE3);
                result.SSSE3 = result.SSE3 && bit_is_open(ecx, CpuFeatureIndex_EAX1_ECX0::SSSE3);
                result.SSE4_1 = result.SSSE3 && bit_is_open(ecx, CpuFeatureIndex_EAX1_ECX0::SSE4_1);
                result.SSE4_2 = result.SSE4_1 && bit_is_open(ecx, CpuFeatureIndex_EAX1_ECX0::SSE4_2);

                // xsave os_xsave
                result.XSAVE = bit_is_open(ecx, CpuFeatureIndex_EAX1_ECX0::XSAVE);
                result.OS_XSAVE = bit_is_open(ecx, CpuFeatureIndex_EAX1_ECX0::OS_XSAVE);

                // 只有在 xsave 和 os_xsave 为 true 的时候，才能进行 xgetbv 检查，AVX指令集才可用
                if (result.XSAVE && result.OS_XSAVE)
                {
                    xcr0 = xgetbv(0);

                    // ------------------------- AVX -------------------------
                    const bool os_support_avx = bit_is_open(xcr0, CpuXSaveStateIndex::SSE) &&
                                                bit_is_open(xcr0, CpuXSaveStateIndex::AVX);

                    result.AVX = result.SSE4_1 && bit_is_open(ecx, CpuFeatureIndex_EAX1_ECX0::AVX) && os_support_avx;
                    result.F16C = result.AVX && bit_is_open(ecx, CpuFeatureIndex_EAX1_ECX0::F16C);
                    result.FMA3 = result.AVX && bit_is_open(ecx, CpuFeatureIndex_EAX1_ECX0::FMA3);
                }
            }

            // ------------------ EAX 7 ECX 0 ------------------
            if (max_leaf >= 7)
            {
                // EAX 7, ECX 0
                cpuid(7, 0, abcd);
                const uint32_t ebx = abcd[1];

                result.AVX2 = result.AVX && bit_is_open(ebx, CpuFeatureIndex_EAX7_ECX0::AVX2);


                // ------------------------- AVX-512 family -------------------------
                const bool os_support_avx_512 = bit_is_open(xcr0, CpuXSaveStateIndex::SSE) &&
                                                bit_is_open(xcr0, CpuXSaveStateIndex::AVX) &&
                                                bit_is_open(xcr0, CpuXSaveStateIndex::AVX_512_K0_K7) &&
                                                bit_is_open(xcr0, CpuXSaveStateIndex::AVX_512_LOW_256) &&
                                                bit_is_open(xcr0, CpuXSaveStateIndex::AVX_512_HIGH_256);

                result.AVX512_F = result.AVX2 && bit_is_open(ebx, CpuFeatureIndex_EAX7_ECX0::AVX_512_F) && os_support_avx_512;
            }

            #elif KSIMD_ARCH_ARM_ANY

            // linux
            #if KSIMD_OS_LINUX

                #if KSIMD_ARCH_ARM_64
                unsigned long hwcaps = getauxval(AT_HWCAP);

                result.NEON = ((hwcaps & HWCAP_ASIMD) != 0);
                result.SVE  = ((hwcaps & HWCAP_SVE) != 0);
                result.ARM_CRC32 = ((hwcaps & HWCAP_CRC32) != 0);

                #elif KSIMD_ARCH_ARM_32
                #error TODO arm32 cpu feature detect.

                #else
                #error unknown arm arch.
                #endif
            #endif // !linux

            // apple
            #if KSIMD_OS_MACOS
                #error TODO: apple arm intrinsic cpu feature detect.
            #endif // mac os

            // windows on arm
            #if KSIMD_OS_WINDOWS
                #error TODO: windows on arm intrinsic cpu feature detect.
            #endif // windows on arm

            #else
            #error unknown arch
            #endif

            return result;
        }();

        return info;
    }
}
// clang-format on
