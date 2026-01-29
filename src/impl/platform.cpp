#include "kSimd/impl/platform.hpp"

#if defined(_MSC_VER)
    #include <intrin.h>
#else
    #include <cpuid.h>
#endif

#include <cstdint>

#include "kSimd/impl/utils.hpp"

KSIMD_NAMESPACE_BEGIN

namespace detail
{
    // 首先需要读取 EAX 1 寄存器，接下来才能用下面的枚举判断
    enum class CpuFeatureIndex_EAX1 : uint32_t
    {
        // see https://en.wikipedia.org/wiki/CPUID

        // ECX 寄存器的 feature
        SSE3        = 0 , // EAX 1, ECX  0
        SSSE3       = 9 , // EAX 1, ECX  9
        FMA3        = 12, // EAX 1, ECX 12
        SSE4_1      = 19, // EAX 1, ECX 19
        SSE4_2      = 20, // EAX 1, ECX 20
        XSAVE       = 26, // EAX 1, ECX 26
        OS_XSAVE    = 27, // EAX 1, ECX 27
        AVX         = 28, // EAX 1, ECX 28
        F16C        = 29, // EAX 1, ECX 29

        // EDX 寄存器的 feature
        FXSR        = 24, // EAX 1, EDX 24
        SSE         = 25, // EAX 1, EDX 25
        SSE2        = 26, // EAX 1, EDX 26
    };

    enum class CpuFeatureIndex_EAX7 : uint32_t
    {
        AVX2        = 5 , // EAX 7, EBX  5
        AVX_512_F   = 16, // EAX 7, EBX 16
    };

    enum class CpuXSaveStateIndex : uint64_t
    {
        // see https://en.wikipedia.org/wiki/CPUID XSAVE State-components

        // bit 1: SSE state: XMM0-XMM15 and MXCSR
        // bit 2: AVX: YMM0-YMM15
        // bit 5: AVX-512: opmask registers k0-k7
        // bit 6: AVX-512: ZMM_Hi256 ZMM0-ZMM15
        // bit 7: AVX-512: Hi16_ZMM ZMM16-ZMM31

        SSE                 = 1 , // XMM0-XMM15 and MXCSR
        AVX                 = 2 , // YMM0-YMM15
        AVX_512_K0_K7       = 5 , // opmask registers k0-k7
        AVX_512_LOW_256     = 6 , // ZMM0-ZMM15
        AVX_512_HIGH_256    = 7 , // ZMM16-ZMM31
    };

    template<typename T, typename U>
    static constexpr bool bit_is_open(T data, U bit_pos) noexcept
    {
        static_assert(sizeof(T) == sizeof(U));

        using Type = underlying_t<U>;
        return (static_cast<Type>(data) & (static_cast<Type>(1) << static_cast<Type>(bit_pos))) != 0;
    }

#if defined(KSIMD_ARCH_X86_ANY)
    // leaf: EAX, sub_leaf: ECX
    static void cpuid(const uint32_t leaf, const uint32_t sub_leaf, uint32_t* abcd)
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

    static uint64_t xgetbv(uint32_t idx)
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

    static const CpuSupportInfo& get_support_info_impl() noexcept
    {
        using namespace detail;

        static CpuSupportInfo result{};

#if defined(KSIMD_ARCH_X86_ANY)
        uint32_t abcd[4]; // eax, ebx, ecx, edx

        cpuid(0, 0, abcd);
        const uint32_t max_leaf = abcd[0];


        // ------------------ EAX 1 ------------------
        if (max_leaf < 1) // 因为要读取EAX 1，所以 max leaf 必须 >= 1
        {
            return result;
        }

        // 查询 EAX 1, ECX 0
        cpuid(1, 0, abcd);
        const uint32_t ecx = abcd[2];
        const uint32_t edx = abcd[3];

        // ------------------------- FXSR -------------------------
        result.FXSR = bit_is_open(edx, CpuFeatureIndex_EAX1::FXSR);
        if (!result.FXSR)
        {
            return result;
        }

        // ------------------------- SSE family -------------------------
        result.SSE = bit_is_open(edx, CpuFeatureIndex_EAX1::SSE);
        result.SSE2 = result.SSE && bit_is_open(edx, CpuFeatureIndex_EAX1::SSE2);
        result.SSE3 = result.SSE2 && bit_is_open(ecx, CpuFeatureIndex_EAX1::SSE3);
        result.SSSE3 = result.SSE3 && bit_is_open(ecx, CpuFeatureIndex_EAX1::SSSE3);
        result.SSE4_1 = result.SSSE3 && bit_is_open(ecx, CpuFeatureIndex_EAX1::SSE4_1);
        result.SSE4_2 = result.SSE4_1 && bit_is_open(ecx, CpuFeatureIndex_EAX1::SSE4_2);


        result.XSAVE = bit_is_open(ecx, CpuFeatureIndex_EAX1::XSAVE);
        result.OS_XSAVE = bit_is_open(ecx, CpuFeatureIndex_EAX1::OS_XSAVE);
        // 只有在 xsave 和 os_xsave 为 true 的时候，才能进行 xgetbv 检查，AVX指令集才可用
        if (!result.XSAVE || !result.OS_XSAVE)
        {
            return result;
        }

        const uint64_t xcr0 = xgetbv(0);

        // ------------------------- AVX family -------------------------
        const bool os_support_avx = bit_is_open(xcr0, CpuXSaveStateIndex::SSE) &&
                                    bit_is_open(xcr0, CpuXSaveStateIndex::AVX);

        result.AVX = result.SSE4_1 && bit_is_open(ecx, CpuFeatureIndex_EAX1::AVX) && os_support_avx;
        result.F16C = result.AVX && bit_is_open(ecx, CpuFeatureIndex_EAX1::F16C);
        result.FMA3 = result.AVX && bit_is_open(ecx, CpuFeatureIndex_EAX1::FMA3);

        // ------------------ EAX 7 ------------------
        if (max_leaf < 7) // 因为要读取EAX 7，所以 max leaf 必须 >= 7
        {
            return result;
        }

        // EAX 7, ECX 0
        cpuid(7, 0, abcd);
        const uint32_t ebx = abcd[1];

        result.AVX2 = result.AVX && bit_is_open(ebx, CpuFeatureIndex_EAX7::AVX2);


        // ------------------------- AVX-512 family -------------------------
        const bool os_support_avx_512 = os_support_avx &&
                                        bit_is_open(xcr0, CpuXSaveStateIndex::AVX_512_K0_K7) &&
                                        bit_is_open(xcr0, CpuXSaveStateIndex::AVX_512_LOW_256) &&
                                        bit_is_open(xcr0, CpuXSaveStateIndex::AVX_512_HIGH_256);

        result.AVX512_F = result.AVX2 && bit_is_open(ebx, CpuFeatureIndex_EAX7::AVX_512_F) && os_support_avx_512;

        return result;
#else
        // arm NEON
#endif
    }

    size_t required_alignment() noexcept
    {
        const auto& supports = get_cpu_support_info();
        if (supports.AVX512_F)
        {
            return Alignment::AVX512_Family;
        }
        if (supports.AVX)
        {
            return Alignment::AVX_Family;
        }
        if (supports.SSE)
        {
            return Alignment::SSE_Family;
        }

        // 标量返回0
        return 0;
    }
}

const CpuSupportInfo& get_cpu_support_info() noexcept
{
    static const CpuSupportInfo& s = detail::get_support_info_impl();
    return s;
}

KSIMD_NAMESPACE_END
