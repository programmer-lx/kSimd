#pragma once

#if defined(__JETBRAINS_IDE__) || defined(__CLION_IDE__)
    #define KSIMD_IDE 1
#endif

#ifdef KSIMD_IDE

    #define KSIMD_IDE_BASE_OP_HINT(op_name, scalar_type) \
        using op_name = KSIMD_NAMESPACE_NAME::BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_FALLBACK, scalar_type>;

    #define KSIMD_IDE_TYPE_OP_HINT(op_name) \
        using op_name = KSIMD_NAMESPACE_NAME::TypeOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_FALLBACK>;

#else

    #define KSIMD_IDE_BASE_OP_HINT(op_name, scalar_type)
    #define KSIMD_IDE_TYPE_OP_HINT(op_name)

#endif
