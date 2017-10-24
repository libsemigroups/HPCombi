LINK.o = $(LINK.cc)  # default linker is c++

CXXFLAGS    = -std=c++14 -g -Wall -mavx -mtune=native -O3 -funroll-loops -flax-vector-conversions
# CXXFLAGS    = -std=c++11 -g -Wall -mavx -mtune=native
# -Winline -fcilkplus 
# -Wall -Wno-missing-braces -Wno-unused-variable  -Wno-attributes

TARGET = cycle sort inverse essai
all: $(TARGET)

testtools.o: testtools.cpp testtools.hpp perm16.hpp perm16.impl power.hpp
perm16.o: perm16.hpp perm16.impl power.hpp
cycle.o: cycle.cpp perm16.hpp perm16.impl power.hpp testtools.hpp
essai.o: essai.cpp perm16.hpp perm16.impl power.hpp testtools.hpp
inverse.o: inverse.cpp perm16.hpp perm16.impl power.hpp testtools.hpp
perm16.o: perm16.cpp perm16.hpp perm16.impl power.hpp
sort.o: sort.cpp perm16.hpp perm16.impl power.hpp testtools.hpp

cycle: cycle.o perm16.o testtools.o
sort: sort.o perm16.o testtools.o
inverse: inverse.o perm16.o testtools.o
essai: essai.o perm16.o testtools.o

clean:
	rm -rf $(TARGET) *.o

lint:
	cpplint.py --filter=-build/namespaces --extensions=hpp,cpp,impl \
	    *.cpp *.hpp *.impl
