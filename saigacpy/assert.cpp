/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saigacpy/assert.h"

#if defined(SAIGA_VULKAN_INCLUDED) || defined(SAIGA_OPENGL_INCLUDED)
#    error This module must be independent of any graphics API.
#endif


namespace Saiga
{
void saiga_assert_fail(const std::string& __assertion, const char* __file, unsigned int __line, const char* __function,
                       const std::string& __message)
{
    std::cout << "Assertion '" << __assertion << "' failed!" << std::endl;
    std::cout << "  File: " << __file << ":" << __line << std::endl;
    std::cout << "  Function: " << __function << std::endl;
    std::cout << "  Message: " << __message << std::endl;

    // stops and raise SIGABRT
    std::abort();
}

}  // namespace Saiga
