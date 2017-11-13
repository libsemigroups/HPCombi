#include <algorithm>
#include <array>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <x86intrin.h>

using namespace std;

/**********************************************************************/
/************** Défnitions des types et convertisseurs ****************/
/**********************************************************************/

/** Variable vectorielle
 * vecteur de 16 byte représentant une permutation
 * supporte les commandees vectorielles du processeur
 **/
using perm = uint8_t __attribute__((vector_size(16), __may_alias__));

/**********************************************************************/
/***************** Fonctions d'affichages *****************************/
/**********************************************************************/

/** Affichage perm
 * Définition de l'opérateur d'affichage << pour le type perm
 **/
ostream &operator<<(ostream &stream, perm const &p) {
  stream << "[" << setw(2) << hex << unsigned(p[0]);
  for (unsigned i = 1; i < 16; ++i)
    stream << "," << setw(2) << hex << unsigned(p[i]) << dec;
  stream << "]";
  return stream;
}

/**********************************************************************/
/****** Permutations Variables globales et fonctions de base **********/
/**********************************************************************/

/** Permutation identité **/
const perm permid{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

/** Permutation décalée d'un cran à gauche **/
const perm decal{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15};

int main() {
  const perm v1{2, 1, 7, 4, 9, 15, 12, 0, 5, 3, 6, 8, 11, 10, 14, 13};
  const perm v2{2, 1, 32, 4, 8, 1, 12, 0, 4, 4, 4, 4, 41, 10, 14, 13};
  perm v3;
  v3 = v1 <= v2;

  cout << v1 << endl;
  cout << v2 << endl;
  cout << v3 << endl;

  int b = _mm_movemask_epi8(v3);
  cout << "Application du masque : on obtient un nombre dont les 1 binaires "
          "représentent les positions non nulles : "
       << hex << unsigned(b) << endl;
  cout << "On compte les 1 avec une opération du processeur" << endl;
  cout << _mm_popcnt_u32(b) << endl;
}
