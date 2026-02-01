#pragma once

#define KSIMD_ENABLE_OPERATORS() \
    using ::KSIMD_NAMESPACE_NAME::operator+;  using ::KSIMD_NAMESPACE_NAME::operator+=; \
    using ::KSIMD_NAMESPACE_NAME::operator-;  using ::KSIMD_NAMESPACE_NAME::operator-=; \
    using ::KSIMD_NAMESPACE_NAME::operator*;  using ::KSIMD_NAMESPACE_NAME::operator*=; \
    using ::KSIMD_NAMESPACE_NAME::operator/;  using ::KSIMD_NAMESPACE_NAME::operator/=; \
    using ::KSIMD_NAMESPACE_NAME::operator&;  using ::KSIMD_NAMESPACE_NAME::operator&=; \
    using ::KSIMD_NAMESPACE_NAME::operator|;  using ::KSIMD_NAMESPACE_NAME::operator|=; \
    using ::KSIMD_NAMESPACE_NAME::operator^;  using ::KSIMD_NAMESPACE_NAME::operator^=; \
    using ::KSIMD_NAMESPACE_NAME::operator~
