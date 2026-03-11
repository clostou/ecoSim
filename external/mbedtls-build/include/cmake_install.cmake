# Install script for directory: E:/worksapce/ecoSim/external/mbedtls-src/include

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "E:/worksapce/ecoSim/out/install/x64-Debug")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/mbedtls" TYPE FILE PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ FILES
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/aes.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/aria.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/asn1.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/asn1write.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/base64.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/bignum.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/block_cipher.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/build_info.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/camellia.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/ccm.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/chacha20.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/chachapoly.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/check_config.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/cipher.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/cmac.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/compat-2.x.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/config_adjust_legacy_crypto.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/config_adjust_legacy_from_psa.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/config_adjust_psa_from_legacy.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/config_adjust_psa_superset_legacy.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/config_adjust_ssl.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/config_adjust_x509.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/config_psa.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/constant_time.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/ctr_drbg.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/debug.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/des.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/dhm.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/ecdh.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/ecdsa.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/ecjpake.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/ecp.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/entropy.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/error.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/gcm.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/hkdf.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/hmac_drbg.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/lms.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/mbedtls_config.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/md.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/md5.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/memory_buffer_alloc.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/net_sockets.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/nist_kw.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/oid.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/pem.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/pk.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/pkcs12.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/pkcs5.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/pkcs7.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/platform.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/platform_time.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/platform_util.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/poly1305.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/private_access.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/psa_util.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/ripemd160.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/rsa.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/sha1.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/sha256.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/sha3.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/sha512.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/ssl.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/ssl_cache.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/ssl_ciphersuites.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/ssl_cookie.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/ssl_ticket.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/threading.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/threading_alt.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/timing.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/version.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/x509.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/x509_crl.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/x509_crt.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/mbedtls/x509_csr.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/psa" TYPE FILE PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ FILES
    "E:/worksapce/ecoSim/external/mbedtls-src/include/psa/build_info.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/psa/crypto.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/psa/crypto_adjust_auto_enabled.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/psa/crypto_adjust_config_dependencies.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/psa/crypto_adjust_config_key_pair_types.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/psa/crypto_adjust_config_synonyms.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/psa/crypto_builtin_composites.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/psa/crypto_builtin_key_derivation.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/psa/crypto_builtin_primitives.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/psa/crypto_compat.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/psa/crypto_config.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/psa/crypto_driver_common.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/psa/crypto_driver_contexts_composites.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/psa/crypto_driver_contexts_key_derivation.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/psa/crypto_driver_contexts_primitives.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/psa/crypto_extra.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/psa/crypto_legacy.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/psa/crypto_platform.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/psa/crypto_se_driver.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/psa/crypto_sizes.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/psa/crypto_struct.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/psa/crypto_types.h"
    "E:/worksapce/ecoSim/external/mbedtls-src/include/psa/crypto_values.h"
    )
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "E:/worksapce/ecoSim/external/mbedtls-build/include/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
