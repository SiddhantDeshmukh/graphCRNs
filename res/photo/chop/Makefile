#FC        = g77
#FC        = pgf90
#FC        = ifc
FC         = gfortran
FFLAGS     = -O3 -fpic 
LDFLAGS    = -shared -O3 
CC         = gcc
CFLAGS     = -O3 -fPIC -m64 
#
#CFLAGS     = -O0 -fPIC -m64
#FFLAGS     = -O0 -CB -fpe0 -fpic
#CFLAGS     = -O -fPIC -m64
#FFLAGS     = -O -CB -fpe0 -fpic
#LDFLAGS    = -shared -O -CB -fpe0 
#CFLAGS     = -O -fPIC -m64
#FFLAGS     = -O -fpe0 -fpic
#LDFLAGS    = -shared -O -fpe0 
#CFLAGS    = -g -fPIC -m64
#FFLAGS    = -g -C -fpe1 -fpic
#LDFLAGS   = -shared -g -C -fpe1

# === Suffix handling =========================================================

# --- Suffixes ---
.SUFFIXES:
.SUFFIXES: .o .c .f90 .f

# --- Suffix rules ---
.c.o:
	$(CC) $(CFLAGS) -c $<
.f90.o:
	$(FC) $(FFLAGS) -c $<
.f.o:
	$(FC) $(FFLAGS) -c $<

all: chop_idl.so ohop_idl.so 

chop_idl.so: chop_idl.o chop.o   
	$(FC) $(LDFLAGS) chop_idl.o chop.o -o $@
	mv chop_idl.so ${IDL_SO}

ohop_idl.so: ohop_idl.o ohop.o   
	$(FC) $(LDFLAGS) ohop_idl.o ohop.o -o $@
	mv ohop_idl.so ${IDL_SO}

clean:
	rm *.o *.mod

 

