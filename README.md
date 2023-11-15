# HPCombi
High Performance Combinatorics in C++ using vector instructions v0.0.8

HPCombi is a C++17 header-only library using the SSE and AVX instruction sets,
and some equivalents, for very fast manipulation of combinatorial
objects such as transformations, permutations, and boolean matrices of small
size. The goal of this project is to implement various new algorithms and
benchmark them on various compiler and architectures.

HPCombi was initially designed using the SSE and AVX instruction sets, and did
not work on machines without these instructions (such as ARM). From v1.0.0
HPCombi supports processors with other instruction sets also, via
[simd-everywhere](https://github.com/simd-everywhere/simde). It might be the
case that the greatest performance gains are achieved on processors supporting
the SSE and AVX instruction sets, but the HPCombi benchmarks indicate that
there are also still signficant gains on other processors too.
<!-- TODO add link to HPCombi wiki with benchmark graphs -->

## Authors

- Florent Hivert <florent.hivert@lisn.fr>
- James Mitchell <jdm3@st-andrews.ac.uk>

## Contributors

- Reinis Cirpons : CI + benchmark graphs
- Viviane Pons : discussions about algorithms
- Finn Smith : discussions + `BMat8` reference code

## Thanks

- The development of HPCombi was partly funded by the
  [OpenDreamKit](http://opendreamkit.org/) Horizon 2020 European Research
  Infrastructure project (#676541), which the authors acknowledge with thanks.
- Thanks also to the
  [simd-everywhere](https://github.com/simd-everywhere/simde) and
  [catch2](https://github.com/catchorg/Catch2) authors and contributors for
  their excellent libraries!
