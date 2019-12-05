#include <stdio.h>
#include "egrss.h"

/**
\author Martin S. Andersen
\copyright BSD-2-clause

\brief Prints error message to \c stderr.
*/
void egrss_err_param(const char * func, int info) {
  fprintf(stderr,"** On entry to %s, parameter number %d had an illegal value\n", func, -info);
  return;
}
