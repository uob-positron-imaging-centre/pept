/**
 * File   : binary_converter_ext.h
 * License: GNU v3.0
 * Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
 * Date   : 29.03.2021
 */

#ifndef BINARY_CONVERTER_EXT
#define BINARY_CONVERTER_EXT

#include <sys/types.h>

double* read_adac_binary(const char *filepath, ssize_t *lors_elements);

#endif
