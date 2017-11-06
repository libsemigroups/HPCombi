LINK.o = $(LINK.cc)  # default linker is c++

CXXFLAGS    = -std=c++14 -g -Wall -mavx -mtune=native -O3 -funroll-loops -flax-vector-conversions
# CXXFLAGS    = -std=c++11 -g -Wall -mavx -mtune=native
# -Winline -fcilkplus 
# -Wall -Wno-missing-braces -Wno-unused-variable  -Wno-attributes

TARGET = cycle sort length inverse essai
all: $(TARGET)

testtools.o: testtools.cpp testtools.hpp perm16.hpp perm16_impl.hpp power.hpp
perm16.o: perm16.cpp perm16.hpp perm16_impl.hpp power.hpp

cycle.o: cycle.cpp perm16.hpp perm16_impl.hpp power.hpp testtools.hpp
cycle: cycle.o perm16.o testtools.o

sort.o: sort.cpp perm16.hpp perm16_impl.hpp power.hpp testtools.hpp
sort: sort.o perm16.o testtools.o

length.o: length.cpp perm16.hpp perm16_impl.hpp power.hpp testtools.hpp
length: length.o perm16.o testtools.o

inverse.o: inverse.cpp perm16.hpp perm16_impl.hpp power.hpp testtools.hpp
inverse: inverse.o perm16.o testtools.o

essai.o: essai.cpp perm16.hpp perm16_impl.hpp power.hpp testtools.hpp
essai: essai.o perm16.o testtools.o


clean:
	rm -rf $(TARGET) *.o

lint:
	cpplint.py --filter=-build/namespaces --extensions=hpp,cpp *.cpp *.hpp
