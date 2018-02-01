#ifndef BENCH_FIXTURE
#define BENCH_FIXTURE

//~ #include <benchmark/benchmark.h>
#include "perm16.hpp"
#include "perm_generic.hpp"
#include "testtools.hpp"

using HPCombi::epu8;
using HPCombi::Vect16;
using HPCombi::PTransf16;
using HPCombi::Transf16;
using HPCombi::Perm16;
using HPCombi::VectGeneric;

typedef VectGeneric<1024, uint16_t> Vect1024;
typedef VectGeneric<2048, uint16_t> Vect2048;
typedef VectGeneric<8192, uint16_t> Vect8192;
typedef VectGeneric<32768, uint16_t> Vect32768;
typedef VectGeneric<131072, uint32_t> Vect131072;
constexpr uint_fast64_t number = 1;
constexpr uint_fast64_t repeat = 1000;


inline static void escape(void *p) {
  asm volatile("" : : "g"(p) : "memory");
}

inline static void clobber() {
  asm volatile("" : : : "memory");
}



class Fix_perm16 {
public :
  Fix_perm16() :
          PPa({1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}),
          PPb({1, 2, 3, 6, 0, 5, 4, 7, 8, 9, 10, 11, 12, 15, 14, 13}),
		  sample(HPCombi::rand_perms(number))
          {}
		  
  ~Fix_perm16() {}
  const Perm16 PPa, PPb;
  const std::vector<Perm16> sample;
};


class Fix_transf16 {
public :
  Fix_transf16() : 
		  zero(Vect16({}, 0)),
          P01(Vect16({0, 1}, 0)),
          P10(Vect16({1, 0}, 0)),
          P11(Vect16({1, 1}, 0)),
          P1(Vect16({}, 1)),
          RandT({3, 1, 0, 14, 15, 13, 5, 10, 2, 11, 6, 12, 7, 4, 8, 9})
          {}
		  
  ~Fix_transf16() {}
  
  const Transf16 zero, P01, P10, P11, P1, RandT;
};

class Fix_generic {
public :
  Fix_generic() :          
          id(Vect1024(0, 0)), // random permutation
          randShuf(Vect1024(0, -1)), // random permutation
          rand(Vect1024(0, -2)), // Randdom transformation
          zeros(Vect1024(0)), // only zeros
          sample1024({id, randShuf, rand, zeros}),
          sample2048({id, randShuf, rand, zeros}),
          sample8192({id, randShuf, rand, zeros}),
          sample32768({id, randShuf, rand, zeros}),
          sample131072({id, randShuf, rand, zeros})
          {}
		  
  ~Fix_generic() {}

  const Vect1024 id, randShuf, rand, zeros;
  const std::vector<Vect1024> sample1024, sample2048, sample8192, sample32768, sample131072;
};



#endif  // BENCH_FIXTURE
