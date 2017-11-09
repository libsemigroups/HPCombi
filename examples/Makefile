LINK.o = $(LINK.cc)  # default linker is c++

CXXFLAGS    = -std=c++14 -O3 -g -Wall -mavx -mtune=native -funroll-loops -flax-vector-conversions
# CXXFLAGS    = -std=c++11 -g -Wall -mavx -mtune=native
# -Winline -fcilkplus 
# -Wall -Wno-missing-braces -Wno-unused-variable  -Wno-attributes

TARGET = Renner RD Trans
all: $(TARGET)

perm16.o: perm16.cpp perm16.hpp perm16_impl.hpp power.hpp
RD.o: RD.cpp perm16.hpp perm16_impl.hpp power.hpp
RD: RD.o perm16.o
Trans.o: Trans.cpp perm16.hpp perm16_impl.hpp power.hpp
Trans: Trans.o perm16.o
Renner.o: Renner.cpp perm16.hpp perm16_impl.hpp power.hpp
Renner: Renner.o perm16.o

clean:
	rm -rf $(TARGET) *.o

lint:
	cpplint.py --filter=-build/namespaces --extensions=hpp,cpp *.cpp *.hpp
