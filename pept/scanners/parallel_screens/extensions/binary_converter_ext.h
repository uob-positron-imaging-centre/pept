/**
 * File   : binary_converter_ext.h
 * License: GNU v3.0
 * Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
 * Date   : 29.03.2021
 */

#ifndef BINARY_CONVERTER_EXT
#define BINARY_CONVERTER_EXT


#if defined(_MSC_VER)
	// Support the bloody unconforming mess that MSVC is; allow using fopen and ssize_t
	#define _CRT_SECURE_NO_DEPRECATE
	#include <BaseTsd.h>
	typedef SSIZE_T ssize_t;
#else
	#include <sys/types.h>
#endif


double* read_adac_binary(const char* filepath, ssize_t* lors_elements);

#endif