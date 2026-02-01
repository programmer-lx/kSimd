#pragma once

#define KSIMD_ENABLE_OPERATORS() \
    using ::ksimd::operator+;  using ::ksimd::operator+=; \
    using ::ksimd::operator-;  using ::ksimd::operator-=; \
    using ::ksimd::operator*;  using ::ksimd::operator*=; \
    using ::ksimd::operator/;  using ::ksimd::operator/=; \
    using ::ksimd::operator&;  using ::ksimd::operator&=; \
    using ::ksimd::operator|;  using ::ksimd::operator|=; \
    using ::ksimd::operator^;  using ::ksimd::operator^=; \
    using ::ksimd::operator~
