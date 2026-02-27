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
#include <cstring> // std::memcpy

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
        AES_NI      = 25, // EAX 1 ECX 0, ECX 25
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
        // EBX
        AVX2                = 5 , // EAX 7 ECX 0, EBX  5
        AVX512_F            = 16, // EAX 7 ECX 0, EBX 16
        AVX512_DQ           = 17, // EAX 7 ECX 0, EBX 17
        AVX512_IFMA         = 21, // EAX 7 ECX 0, EBX 21
        AVX512_CD           = 28, // EAX 7 ECX 0, EBX 28
        SHA                 = 29, // EAX 7 ECX 0, EBX 29
        AVX512_BW           = 30, // EAX 7 ECX 0, EBX 30
        AVX512_VL           = 31, // EAX 7 ECX 0, EBX 31

        // ECX
        AVX512_VBMI         = 1 , // EAX 7 ECX 0, ECX  1
        AVX512_VBMI2        = 6 , // EAX 7 ECX 0, ECX  6
        AVX512_VNNI         = 11, // EAX 7 ECX 0, ECX 11
        AVX512_BITALG       = 12, // EAX 7 ECX 0, ECX 12
        AVX512_VPOPCNTDQ    = 14, // EAX 7 ECX 0, ECX 14

        // EDX
        AVX512_VP2INTERSECT = 8 , // EAX 7 ECX 0, EDX  8
        AVX512_FP16         = 23, // EAX 7 ECX 0, EDX 23
    };

    enum class CpuFeatureIndex_EAX7_ECX1 : uint32_t
    {
        // EAX
        SHA512          = 0 , // EAX 7 ECX 1, EAX  0
        SM3             = 1 , // EAX 7 ECX 1, EAX  1
        SM4             = 2 , // EAX 7 ECX 1, EAX  2
        AVX_VNNI        = 4 , // EAX 7 ECX 1, EAX  4
        AVX512_BF16     = 5 , // EAX 7 ECX 1, EAX  5
        AVX_IFMA        = 23, // EAX 7 ECX 1, EAX 23

        // EDX
        AVX_VNNI_INT8   = 4 , // EAX 7 ECX 1, EDX  4
        AVX_NE_CONVERT  = 5 , // EAX 7 ECX 1, EDX  5
        AVX_VNNI_INT16  = 10, // EAX 7 ECX 1, EDX 10
    };

    enum class CpuXSaveStateIndex : uint64_t
    {
        // see https://en.wikipedia.org/wiki/CPUID XSAVE State-components

        XMM             = 1 , // XMM0-XMM15 and MXCSR
        YMM             = 2 , // YMM0-YMM15
        K0_K7           = 5 , // opmask registers k0-k7
        ZMM_LOW_256     = 6 , // ZMM0-ZMM15
        ZMM_HIGH_256    = 7 , // ZMM16-ZMM31
    };

    // leaf: EAX, sub_leaf: ECX
    void cpuid(const uint32_t leaf, const uint32_t sub_leaf, uint32_t* abcd)
    {
        #if defined(_MSC_VER)

        int regs[4];
        __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(sub_leaf));
        for (int i = 0; i < 4; ++i)
        {
            abcd[i] = static_cast<uint32_t>(regs[i]);
        }

        #else

        uint32_t a;
        uint32_t b;
        uint32_t c;
        uint32_t d;
        __cpuid_count(leaf, sub_leaf, a, b, c, d);
        abcd[0] = a;
        abcd[1] = b;
        abcd[2] = c;
        abcd[3] = d;

        #endif
    }

    uint64_t xgetbv(uint32_t idx)
    {
        #if defined(_MSC_VER)

        return _xgetbv(idx);

        #else

        uint32_t eax, edx;
        __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(idx));
        return (static_cast<uint64_t>(edx) << 32) | eax;

        #endif
    }
    #endif // arch x86 any
}

namespace ksimd
{
    const CpuSupportInfo& get_cpu_support_info() noexcept
    {
        #if KSIMD_ARCH_X86_ANY
        static CpuSupportInfo info = []()
        {
            CpuSupportInfo result{};

            uint32_t abcd[4]; // eax, ebx, ecx, edx

            cpuid(0, 0, abcd);
            // 如果 max_leaf == 13，则可以调用 cpuid 查询的EAX的范围是 [0, 13]
            const uint32_t max_leaf = abcd[0];

            uint64_t xcr0 = 0;
            uint32_t eax1_ecx0_edx = 0; // EAX 1, ECX 0 的 EDX 值

            // ------------------ EAX 0 ------------------
            // if (max_leaf >= 0)
            {
                // vendor name
                const uint32_t ebx = abcd[1];
                const uint32_t ecx = abcd[2];
                const uint32_t edx = abcd[3];
                std::memcpy(result.vendor_name, &ebx, sizeof(uint32_t));
                std::memcpy(result.vendor_name + sizeof(uint32_t), &edx, sizeof(uint32_t));
                std::memcpy(result.vendor_name + 2 * sizeof(uint32_t), &ecx, sizeof(uint32_t));
                result.vendor_name[12] = 0;

                // vendor enum
                // Intel "GenuineIntel"
                if (ebx == 0x756e6547 && edx == 0x49656e69 && ecx == 0x6c65746e)
                {
                    result.vendor = CpuVendor::Intel;
                }

                // AMD "AuthenticAMD"
                if (ebx == 0x68747541 && edx == 0x69746e65 && ecx == 0x444d4163)
                {
                    result.vendor = CpuVendor::AMD;
                }
            }

            // ------------------ EAX 1 ------------------
            if (max_leaf >= 1)
            {
                // EAX 1, ECX 0
                cpuid(1, 0, abcd);
                const uint32_t ebx = abcd[1];
                const uint32_t ecx = abcd[2];
                const uint32_t edx = abcd[3];
                eax1_ecx0_edx = edx;

                result.logical_cores = (ebx >> 16) & 0xff; // EBX[23:16]

                // ------------------------- FXSR -------------------------
                result.fxsr = bit_is_open(edx, CpuFeatureIndex_EAX1_ECX0::FXSR);

                // ------------------------- SSE family -------------------------
                result.sse = result.fxsr && bit_is_open(edx, CpuFeatureIndex_EAX1_ECX0::SSE);
                result.sse2 = result.sse && bit_is_open(edx, CpuFeatureIndex_EAX1_ECX0::SSE2);
                result.sse3 = result.sse2 && bit_is_open(ecx, CpuFeatureIndex_EAX1_ECX0::SSE3);
                result.ssse3 = result.sse3 && bit_is_open(ecx, CpuFeatureIndex_EAX1_ECX0::SSSE3);
                result.sse4_1 = result.ssse3 && bit_is_open(ecx, CpuFeatureIndex_EAX1_ECX0::SSE4_1);
                result.sse4_2 = result.sse4_1 && bit_is_open(ecx, CpuFeatureIndex_EAX1_ECX0::SSE4_2);

                // ------------------------- XSAVE -------------------------
                result.xsave = bit_is_open(ecx, CpuFeatureIndex_EAX1_ECX0::XSAVE);
                result.os_xsave = bit_is_open(ecx, CpuFeatureIndex_EAX1_ECX0::OS_XSAVE);
                // 只有在 xsave 和 os_xsave 为 true 的时候，才能进行 xgetbv 检查，AVX指令集才可用
                if (result.xsave && result.os_xsave)
                {
                    xcr0 = xgetbv(0);
                }

                // ------------------------- AVX family -------------------------
                const bool os_support_avx = bit_is_open(xcr0, CpuXSaveStateIndex::XMM) &&
                                            bit_is_open(xcr0, CpuXSaveStateIndex::YMM);

                result.avx = result.sse4_1 && bit_is_open(ecx, CpuFeatureIndex_EAX1_ECX0::AVX) && os_support_avx;
                result.f16c = result.avx && bit_is_open(ecx, CpuFeatureIndex_EAX1_ECX0::F16C);
                result.fma3 = result.avx && bit_is_open(ecx, CpuFeatureIndex_EAX1_ECX0::FMA3);

                // other
                result.aes_ni = bit_is_open(ecx, CpuFeatureIndex_EAX1_ECX0::AES_NI);
                result.popcnt = bit_is_open(ecx, CpuFeatureIndex_EAX1_ECX0::POPCNT);
            }

            // ------------------ EAX 4 ------------------
            if (max_leaf >= 4)
            {
                cpuid(4, 0, abcd);
                const uint32_t eax = abcd[0];

                if (result.vendor == CpuVendor::Intel)
                {
                    result.physical_cores = ((eax >> 26) & 0x3f) + 1; // EAX[31:26] + 1
                }
            }

            // ------------------ EAX 7 ------------------
            if (max_leaf >= 7)
            {
                // EAX 7, ECX 0
                cpuid(7, 0, abcd);
                const uint32_t eax7_subleaf_count = abcd[0];
                {
                    const uint32_t ebx = abcd[1];
                    const uint32_t ecx = abcd[2];
                    const uint32_t edx = abcd[3];

                    result.avx2 = result.avx && bit_is_open(ebx, CpuFeatureIndex_EAX7_ECX0::AVX2);


                    // ------------------------- AVX-512 family -------------------------
                    const bool os_support_avx_512 = bit_is_open(xcr0, CpuXSaveStateIndex::XMM) &&
                                                    bit_is_open(xcr0, CpuXSaveStateIndex::YMM) &&
                                                    bit_is_open(xcr0, CpuXSaveStateIndex::K0_K7) &&
                                                    bit_is_open(xcr0, CpuXSaveStateIndex::ZMM_LOW_256) &&
                                                    bit_is_open(xcr0, CpuXSaveStateIndex::ZMM_HIGH_256);

                    // ebx
                    result.avx512_f = result.avx2 &&
                                      result.fma3 &&
                                      result.f16c &&
                                      bit_is_open(ebx, CpuFeatureIndex_EAX7_ECX0::AVX512_F) &&
                                      os_support_avx_512;
                    result.avx512_bw = result.avx512_f && bit_is_open(ebx, CpuFeatureIndex_EAX7_ECX0::AVX512_BW);
                    result.avx512_cd = result.avx512_f && bit_is_open(ebx, CpuFeatureIndex_EAX7_ECX0::AVX512_CD);
                    result.avx512_dq = result.avx512_f && bit_is_open(ebx, CpuFeatureIndex_EAX7_ECX0::AVX512_DQ);
                    result.avx512_ifma = result.avx512_f && bit_is_open(ebx, CpuFeatureIndex_EAX7_ECX0::AVX512_IFMA);
                    result.avx512_vl = result.avx512_f && bit_is_open(ebx, CpuFeatureIndex_EAX7_ECX0::AVX512_VL);

                    result.sha = bit_is_open(ebx, CpuFeatureIndex_EAX7_ECX0::SHA);

                    // ecx
                    result.avx512_vpopcntdq = result.avx512_f && bit_is_open(ecx, CpuFeatureIndex_EAX7_ECX0::AVX512_VPOPCNTDQ);
                    result.avx512_bitalg = result.avx512_f && bit_is_open(ecx, CpuFeatureIndex_EAX7_ECX0::AVX512_BITALG);
                    result.avx512_vbmi = result.avx512_f && bit_is_open(ecx, CpuFeatureIndex_EAX7_ECX0::AVX512_VBMI);
                    result.avx512_vbmi2 = result.avx512_f && bit_is_open(ecx, CpuFeatureIndex_EAX7_ECX0::AVX512_VBMI2);
                    result.avx512_vnni = result.avx512_f && bit_is_open(ecx, CpuFeatureIndex_EAX7_ECX0::AVX512_VNNI);

                    // edx
                    result.avx512_vp2intersect = result.avx512_f && bit_is_open(edx, CpuFeatureIndex_EAX7_ECX0::AVX512_VP2INTERSECT);
                    result.avx512_fp16 = result.avx512_f && bit_is_open(edx, CpuFeatureIndex_EAX7_ECX0::AVX512_FP16);
                }

                // EAX 7 ECX 1
                if (eax7_subleaf_count >= 1)
                {
                    cpuid(7, 1, abcd);
                    const uint32_t eax = abcd[0];
                    const uint32_t edx = abcd[3];

                    // eax
                    result.avx_vnni = result.avx2 && bit_is_open(eax, CpuFeatureIndex_EAX7_ECX1::AVX_VNNI);
                    result.avx_ifma = result.avx2 && bit_is_open(eax, CpuFeatureIndex_EAX7_ECX1::AVX_IFMA);

                    result.avx512_bf16 = result.avx512_f && bit_is_open(eax, CpuFeatureIndex_EAX7_ECX1::AVX512_BF16);

                    result.sha512 = result.avx2 && bit_is_open(eax, CpuFeatureIndex_EAX7_ECX1::SHA512);
                    result.sm3 = result.avx2 && bit_is_open(eax, CpuFeatureIndex_EAX7_ECX1::SM3);
                    result.sm4 = result.avx2 && bit_is_open(eax, CpuFeatureIndex_EAX7_ECX1::SM4);

                    // edx
                    result.avx_vnni_int8 = result.avx2 && bit_is_open(edx, CpuFeatureIndex_EAX7_ECX1::AVX_VNNI_INT8);
                    result.avx_ne_convert = result.avx2 && bit_is_open(edx, CpuFeatureIndex_EAX7_ECX1::AVX_NE_CONVERT);
                    result.avx_vnni_int16 = result.avx2 && bit_is_open(edx, CpuFeatureIndex_EAX7_ECX1::AVX_VNNI_INT16);
                }
            }

            // ------------------------------------ ext ------------------------------------
            cpuid(0x80000000, 0, abcd);
            const uint32_t max_ext_leaf = abcd[0];

            // ------------------ EAX 0x8000'0008 ------------------
            if (max_ext_leaf >= 0x80000008)
            {
                if (result.vendor == CpuVendor::AMD)
                {
                    cpuid(0x80000008, 0, abcd);
                    const uint32_t ecx = abcd[2];
                    result.physical_cores = (ecx & 0xff) + 1; // ECX[7:0] + 1
                }
            }

            if (max_leaf >= 1)
            {
                result.hyper_threads = (eax1_ecx0_edx & (1 << 28)) && (result.physical_cores < result.logical_cores);
            }

            return result;
        }();
        #elif KSIMD_ARCH_ARM_ANY
        static CpuSupportInfo info = []()
        {
            CpuSupportInfo result{};

            // linux
            #if KSIMD_OS_LINUX
                #if KSIMD_ARCH_ARM_64
                    unsigned long hwcaps = getauxval(AT_HWCAP);

                    // scalar
                    result.arm_scalar_fp = ((hwcaps & HWCAP_FP) != 0);
                    result.arm_scalar_fp16 = ((hwcaps & HWCAP_FPHP) != 0);

                    // NEON
                    result.neon = ((hwcaps & HWCAP_ASIMD) != 0);
                    result.neon_full_fp16 = ((hwcaps & HWCAP_ASIMDHP) != 0);

                    // SVE
                    result.sve = ((hwcaps & HWCAP_SVE) != 0);


                    // other
                    result.arm_crc32 = ((hwcaps & HWCAP_CRC32) != 0);

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

            return result;
        }();
        #else
        #error unknown arch
        #endif

        return info;
    }
}
// clang-format on
