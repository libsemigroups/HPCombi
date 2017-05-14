# Main Makefile for Florent Hivert "Perm16" library
# Copyright (C) 2016 Florent Hivert.

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2, or (at your option)
# any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

SHELL = /bin/sh
top_srcdir = .
srcdir = .


.SUFFIXES:
.SUFFIXES: .c .o

OPT=-g -O

AR = ar
AR_FLAGS = rc
RANLIB = @RANLIB@

CC = @CC@
CFLAGS = -I. @CFLAGS@
LDFLAGS = 
LIBS = 
INSTALL = /usr/bin/install -c
prefix = /usr/local
exec_prefix = ${prefix}
bindir = $(exec_prefix)/bin
libdir = $(prefix)/lib
infodir = $(prefix)/info

SOURCES=cycle.cpp  perm16.cpp  sort.cpp  testtools.cpp
DOCS=PROG.texi PROG.info
MISC=configure mkinstalldirs install-sh aclocal.m4
OBJS=cycle.o  perm16.o  sort.o  testtools.o
LIB_OBJS=

all: cycle sort

# ??? here I make the bindir, libdir and infodir directories; you
# might not need all of these.  also, I assumed the names PROG and
# libMYPROG.a for the program and library.
install: all
	$(top_srcdir)/mkinstalldirs $(bindir)
	$(top_srcdir)/mkinstalldirs $(libdir)
	$(top_srcdir)/mkinstalldirs $(infodir)
	$(INSTALL) PROG $(bindir)/PROG
	$(INSTALL) libMYPROG.a $(libdir)/libMYPROG.a
	$(INSTALL) PROG.info $(infodir)/PROG.info

uninstall:
	/bin/rm -f $(bindir)/PROG
	/bin/rm -f $(libdir)/libMYPROG.a
	/bin/rm -f $(infodir)/PROG.info

libMYPROG.a: $(OBJS)
	/bin/rm -f libMYPROG.a
	$(AR) $(AR_FLAGS) libMYPROG.a $(LIB_OBJS)
	$(RANLIB) libMYPROG.a

PROG: $(OBJS) libMYPROG.a
	$(CC) $(CFLAGS) -o PROG $(OBJS) $(LIBS)

clean:
	/bin/rm -f core *.o $(OBJS) $(LIB_OBJS) libMYPROG.a

distclean: clean
	/bin/rm -f Makefile config.h config.status config.cache config.log \
		marklib.dvi

mostlyclean: clean

maintainer-clean: clean

PROG.info: PROG.texi
	makeinfo PROG.texi

# automatic re-running of configure if the configure.in file has changed
#${srcdir}/configure: configure.in aclocal.m4
#	cd ${srcdir} && autoconf

# autoheader might not change config.h.in, so touch a stamp file
#${srcdir}/config.h.in: stamp-h.in
#${srcdir}/stamp-h.in: configure.in aclocal.m4
#		cd ${srcdir} && autoheader
#		echo timestamp > ${srcdir}/stamp-h.in

config.h: stamp-h
stamp-h: config.h.in config.status
	./config.status
Makefile: Makefile.in config.status
	./config.status
config.status: configure
	./config.status --recheck


