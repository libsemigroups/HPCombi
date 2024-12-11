//****************************************************************************//
//     Copyright (C) 2016-2024 Florent Hivert <Florent.Hivert@lisn.fr>,       //
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

#ifndef HPCOMBI_HPCOMBI_HPP_
#define HPCOMBI_HPCOMBI_HPP_

#include "bmat8.hpp"
#include "debug.hpp"
#include "epu8.hpp"
#include "perm16.hpp"
#include "perm_generic.hpp"
#include "power.hpp"
#include "vect16.hpp"
#include "vect_generic.hpp"

#endif  // HPCOMBI_HPCOMBI_HPP_

/*! \mainpage HPCombi

\section readme_sec Readme

You might want to have a look at [the Readme in the sources](https://github.com/libsemigroups/HPCombi/blob/main/README.md).

\section sec_philo Philosophy
This library provides high performance computations in combinatorics (hence its name).
In practice we observe large speedups in several enumeration problems.

The main idea of the library is a way to encode data as a small sequence of small integers,
that can be handled efficiently by a creative use of vector instructions.
For example, on the current x86 machines, small permutations (N ≤ 16) are very well handled.
Indeed thanks to machine instructions such as PSHUFB (Packed SHUFfle Bytes),
applying a permutation on a vector only takes a few CPU cycles.

Further ideas are:
- Vectorization (MMX, SSE, AVX instructions sets) and careful memory alignment,
- Careful memory management: avoiding all dynamic allocation during the computation,
- Avoid all unnecessary copies (often needed to rewrite the containers),
- Due to combinatorial explosion, sets often don’t fit in the computer’s memory or disks and are enumerated on the fly.

Here are some examples,
the speedup is in comparison to an implementation without vector instructions:

Operation |   Speedup
----------|-----------
Inverting a permutation | 1.28
Sorting a list of bytes | 21.3
Number of cycles of a permutation |  41.5
Number of inversions of a permutation  | 9.39
Cycle type of a permutation | 8.94



\section sec_tips Tips to the user

There is no parallelisation here. To use parallelism with this lib, see for instance:
- Florent Hivert, High Performance Computing Experiments in Enumerative and Algebraic Combinatorics
([pdf](https://plouffe.fr/OEIS/citations/3115936.3115938.pdf), [DOI](https://dx.doi.org/10.1145/3115936.3115938)).
- [OpenCilk](https://github.com/OpenCilk/) or look for another work stealing framework.

Note that memory access can become a problem. It you store many things, most of the time will be spent in fetching from RAM, not computing.
Data structure should preserve locality. You might want to compute some stats on data structure usage and write custom ones.

This lib is implemented with speed in mind, not code safety.
Eg. there are no checks when building a permutation, which could be invalid (like non injective).

We now suggest to have a look, in the menus above, at Classes → [Class list](annotated.html).
*/