#ifndef BENCH_FIXTURE
#define BENCH_FIXTURE

#include "perm_generic.hpp"
#include <vector>

using HPCombi::VectGeneric;

typedef VectGeneric<1024, uint16_t> Vect1024;
typedef VectGeneric<2048, uint16_t> Vect2048;
typedef VectGeneric<8192, uint16_t> Vect8192;
typedef VectGeneric<32768, uint16_t> Vect32768;
typedef VectGeneric<131072, uint32_t> Vect131072;
constexpr uint_fast64_t number = 10000;
constexpr uint_fast64_t repeat = 100;


inline static void escape(void *p) {
  asm volatile("" : : "g"(p) : "memory");
}

inline static void clobber() {
  asm volatile("" : : : "memory");
}


class Fix_generic {
public :
  Fix_generic() :
          id(Vect1024(0, 0)), // random permutation
          randShuf(Vect1024(0, -1)), // random permutation
          rand(Vect1024(0, -2)), // Randdom transformation
          zeros(Vect1024(0)), // only zeros
          sample1024({id, randShuf, rand, zeros}),
      
          id131072(Vect131072(0, 0)), // random permutation
          randShuf131072(Vect131072(0, -1)), // random permutation
          rand131072(Vect131072(0, -2)), // Randdom transformation
          zeros131072(Vect131072(0)), // only zeros
          sample131072({id131072, randShuf131072, rand131072, zeros131072})
          //~ sample( HPCombi::rand_perms2<1024, uint16_t>(number) )
          {}
		  
  ~Fix_generic() {}

  const Vect1024 id, randShuf, rand, zeros;
  const Vect131072 id131072, randShuf131072, rand131072, zeros131072;
  const std::vector<Vect1024> sample1024;
  const std::vector<Vect131072> sample131072;
};



#endif  // BENCH_FIXTURE
