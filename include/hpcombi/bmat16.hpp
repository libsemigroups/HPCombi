//****************************************************************************//
//    Copyright (C) 2018-2024 Finn Smith <fls3@st-andrews.ac.uk>              //
//    Copyright (C) 2018-2024 James Mitchell <jdm3@st-andrews.ac.uk>          //
//    Copyright (C) 2018-2024 Florent Hivert <Florent.Hivert@lisn.fr>,        //
//                                                                            //
//  This file is part of HP-Combi <https://github.com/libsemigroups/HPCombi>  //
//                                                                            //
//  HP-Combi is free software: you can redistribute it and/or modify it       //
//  under the terms of the GNU General Public License as published by the     //
//  Free Software Foundation, either version 3 of the License, or             //
//  (at your option) any later version.                                       //
//                                                                            //
//  HP-Combi is distributed in the hope that it will be useful, but WITHOUT   //
//  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or     //
//  FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License      //
//  for  more details.                                                        //
//                                                                            //
//  You should have received a copy of the GNU General Public License along   //
//  with HP-Combi. If not, see <https://www.gnu.org/licenses/>.               //
//****************************************************************************//

// This file contains a declaration of fast boolean matrices up to dimension 16.

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

#include "simde/x86/avx2.h"

namespace HPCombi {
using xpu16 = uint16_t __attribute__((vector_size(32)));
using xpu64 = uint64_t __attribute__((vector_size(32)));

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
 public:
    xpu64 _data;

    //! A default constructor.
    //!
    //! This constructor gives no guarantees on what the matrix will contain.
    BMat16() noexcept = default;

    //! A constructor.
    //!
    //! This constructor initializes a BMat16 with a 256-bit register
    //! The rows are equal to the 16 chunks, of 16 bits each, 
    //! of the binary representation of the matrix
    explicit BMat16(xpu64 mat) noexcept : 
        _data{mat} {}

    //! A constructor.
    //!
    //! This constructor initializes a matrix with 4 64 bits unsigned int
    explicit BMat16(uint64_t n0, uint64_t n1, uint64_t n2, uint64_t n3) noexcept;

    //! A constructor.
    //!
    //! This is the copy constructor.
    BMat16(BMat16 const &) noexcept = default;

    //! A constructor.
    //!
    //! This is the move constructor.
    BMat16(BMat16 &&) noexcept = default;

    //! A constructor.
    //!
    //! This is the copy assignment constructor.
    BMat16 &operator=(BMat16 const &) noexcept = default;

    //! A constructor.
    //!
    //! This is the move assignment constructor.
    BMat16 &operator=(BMat16 &&) noexcept = default;

    //! A default destructor.
    ~BMat16() = default;

    //! Returns \c true if \c this equals \p that.
    //!
    //! This method checks the mathematical equality of two BMat8 objects.
    bool operator==(BMat16 const &that) const noexcept;

    // //! Returns \c true if \c this does not equal \p that
    // //!
    // //! This method checks the mathematical inequality of two BMat8 objects.
    // bool operator!=(BMat16 const &that) const noexcept { // A changer
    //     return _data != that._data;
    // }

    // Conversion of type of storage, from blocks to lines
    BMat16 to_line() const;

    // Conversion of type of storage, from lines to blocks
    BMat16 to_block() const;

    // //! Returns the entry in the (\p i, \p j)th position.
    // //!
    // //! This method returns the entry in the (\p i, \p j)th position.
    // //! Note that since all matrices are internally represented as 16 x 16, it
    // //! is possible to access entries that you might not believe exist.
    // bool operator()(size_t i, size_t j) const noexcept;

    // //! Sets the (\p i, \p j)th position to \p val.
    // //!
    // //! This method sets the (\p i, \p j)th entry of \c this to \p val.
    // //! Uses the bit twiddle for setting bits found
    // //! <a href=http://graphics.stanford.edu/~seander/bithacks>here</a>.
    // void set(size_t i, size_t j, bool val) noexcept;

    // //! Returns the array representation of \c this.
    // //!
    // //! Returns a two dimensional 8 x 8 array representing the matrix.
    // std::array<std::array<bool, 8>, 8> to_array() const noexcept;

    //! Returns the bitwise or between \c this and \p that
    //!
    //! This method perform the bitwise operator on the matrices and
    //! returns the result as a BMat8
    BMat16 operator|(BMat16 const& that) const noexcept {
        return BMat16(_data | that._data);
    }

    //! Returns the transpose of \c this.
    //!
    //! Returns the standard matrix transpose of a BMat8.
    //! Uses a naive technique, by simply iterating through all entries
    BMat16 transpose_block() const noexcept;

    //! Returns the transpose of \c this
    //!
    //! Returns the standard matrix transpose of a BMat8.
    //! Uses the technique found in Knuth AoCP Vol. 4 Fasc. 1a, p. 15.
    BMat16 transpose() const noexcept {
        return to_block().transpose_block().to_line();
    }

    //! Returns the matrix product of \c this and the transpose of \p that
    //!
    //! This method returns the standard matrix product (over the
    //! boolean semiring) of two BMat8 objects. This is faster than transposing
    //! that and calling the product of \c this with it. Implementation uses
    //! vector instructions.
    BMat16 mult_transpose(BMat16 const &that) const noexcept;

    //! Returns the matrix product of \c this and \p that
    //!
    //! This method returns the standard matrix product (over the
    //! boolean semiring) of two BMat8 objects. This is a fast implementation
    //! using transposition and vector instructions.
    BMat16 operator*(BMat16 const &that) const noexcept {
        return mult_transpose(that.transpose());
    }

    // //! Returns the matrix product of \c this and \p that
    // //!
    // //! This method returns the standard matrix product (over the
    // //! boolean semiring) of two BMat8 objects. It performs the most naive approch
    // //! by simply iterating through all entries using the acces oeprator of BMat8
    // BMat16 mult_naive(BMat16 const& that) const noexcept;

    // //! Returns the matrix product of \c this and \p that
    // //!
    // //! This method returns the standard matrix product (over the
    // //! boolean semiring) of two BMat8 objects. It performs the most naive approch
    // //! by simply iterating through all entries using array conversion.
    // BMat16 mult_naive_array(BMat16 const& that) const noexcept;

    // //! Returns the number of non-zero rows of \c this
    // size_t nr_rows() const noexcept;

    // //! Returns a \c std::vector for rows of \c this
    // // Not noexcept because it constructs a vector
    // std::vector<uint8_t> rows() const;

    // //! Returns the identity BMat8
    // //!
    // //! This method returns the 8 x 8 BMat8 with 1s on the main diagonal.
    // static BMat8 one(size_t dim = 8) noexcept {
    //     HPCOMBI_ASSERT(dim <= 8);
    //     static std::array<uint64_t, 9> const ones = {
    //         0x0000000000000000, 0x8000000000000000, 0x8040000000000000,
    //         0x8040200000000000, 0x8040201000000000, 0x8040201008000000,
    //         0x8040201008040000, 0x8040201008040200, 0x8040201008040201};
    //     return BMat8(ones[dim]);
    // }

    //! Returns a random BMat8
    //!
    //! This method returns a BMat8 chosen at random.
    // Not noexcept because random things aren't
    static BMat16 random();

    // //! Returns a random square BMat8 up to dimension \p dim.
    // //!
    // //! This method returns a BMat8 chosen at random, where only the
    // //! top-left \p dim x \p dim entries may be non-zero.
    // // Not noexcept because BMat8::random above is not
    // static BMat8 random(size_t dim);

    // void swap(BMat16 &that) noexcept { std::swap(this->_data, that._data); }

    //! Write \c this on \c os
    // Not noexcept
    // std::ostream &write(std::ostream &os) const;
};

}  // namespace HPCombi

#include "bmat16_impl.hpp"

#endif