/*#include <stdio.h>
  #include "idl_export.h" */         /* IDL external definitions */

long chop_idl( argc, argv)
int argc;
void *argv[];
{
void chop_();
       chop_(argv[0], argv[1], argv[2], argv[3]);
       return (long) argc;
}

