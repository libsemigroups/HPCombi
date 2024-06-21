#ifndef HPCOMBI_BMAT16_HPP_
#define HPCOMBI_BMAT16_HPP_

#include <array>       // for array
#include <bitset>      // for bitset
#include <cstddef>     // for size_t
#include <cstdint>     // for uint64_t, uint8_t
#include <functional>  // for hash, __scalar_hash
#include <iostream>    // for ostream
#include <memory>      // for hash
#include <utility>     // for pair, swap
#include <vector>      // for vector

#include "debug.hpp"   // for HPCOMBI_ASSERT
#include "epu8.hpp"    // for epu8
#include "perm16.hpp"  // for Perm16

namespace HPCombi {

//! Class for fast boolean matrices of dimension up to 16 x 16
//!
//! The methods for these small matrices over the boolean semiring
//! are more optimised than the generic methods for boolean matrices.
//! Note that all BMat16 are represented internally as an 16 x 16 matrix;
//! any entries not defined by the user are taken to be 0. This does
//! not affect the results of any calculations.
//!
//! BMat16 is a trivial class.
class BMat16 {


    






};

}  // namespace HPCombi

#endif