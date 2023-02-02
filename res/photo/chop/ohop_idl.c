/*#include <stdio.h>
  #include "idl_export.h" */         /* IDL external definitions */

long ohop_idl( argc, argv)
int argc;
void *argv[];
{
void ohop_();
       ohop_(argv[0], argv[1], argv[2], argv[3]);
       return (long) argc;
}

