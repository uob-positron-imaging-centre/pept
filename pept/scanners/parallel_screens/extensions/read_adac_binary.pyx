#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#    pept is a Python library that unifies Positron Emission Particle
#    Tracking (PEPT) research, including tracking, simulation, data analysis
#    and visualisation tools
#
#    Copyright (C) 2019 Sam Manger
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https: // www.gnu.org/licenses/>.


# File              : read_adac_binary.pyx
# License           : License: GNU v3.0
# Author            : Sam Manger <s.manger@bham.ac.uk>
# Date              : 05.11.2019


#!python
#cython: language_level=3

import time

cpdef read_adac_binary(fname, saveas=None, separation=None):
	
	cdef double t0 = time.process_time()


	cdef int GotXA = 0
	cdef int GotYA = 0
	cdef int GotYB = 0
	cdef int GotXB = 0

	cdef int GotA  = 0
	cdef int GotB  = 0

	cdef double TimeStep = 31.25
	cdef double itime = 0
	cdef double itag = 0
	cdef double itagold = 0
	cdef int nevent = 0

	cdef int Gx1=590
	cdef int Gx2=590
	cdef int Gy1=590
	cdef int Gy2=590

	cdef unsigned int word
	cdef unsigned char word3
	cdef unsigned char word2
	cdef unsigned char word1
	cdef unsigned char word0

	cdef double ix
	cdef double ix1
	cdef double ix2
	cdef double iy
	cdef double iy1
	cdef double iy2

	LORs = []

	if saveas == None:
		saveas = fname[:-5]+".csv"

	# cdef char* fname_C = fname
	# cdef char* saveas_C = saveas

	# read_adac_binary_ext(fname_C, saveas_C)

	f_out = open(saveas, "w")

	with open(fname,"rb") as f:

		if separation == None:
			separation = int.from_bytes(f.read(2),'little')
		else:
			f.seek(2)
		gantry_angle = int.from_bytes(f.read(2),'little')
		header = f.seek(1000) # skip the header

		# Read data words until LOR is complete


		word_bytes = f.read(4)

		# while(word!=:

		while(  word_bytes != b'' ):
			# while(GotA and GotB and GotXA and GotXB and GotYA and GotYB) == 0:

			word_bytes = f.read(4)
			word = int.from_bytes(word_bytes,'little')

			if hex(word) == '0xfaceface':
				# print(hex(word))
				# print("Skipped that one\n")
				word_bytes = f.read(4)
				word = int.from_bytes(word_bytes,'little')


			word = word&0xFFFFFFA0

			itag = ((word&0xFF000000) >> 24)
			dtime = itag - itagold
			itagold = itag

			if(dtime<0):
				dtime = dtime + 256
			if(dtime<130):
				itime = itime + (dtime / TimeStep)
			
			prev_word = word

			word3 = (word&0xFF000000) >> 24 
			word2 = (word&0x00FF0000) >> 16 
			word1 = (word&0x0000FF00) >> 8 
			word0 = (word&0x000000FF) 

			# parse data value from inverted and odd order word
			value = 16383 - ((word2&16) // 16) - ((word2&64) // 32) - ((word2&2)*2) - ((word1&128) // 16) - ((word1&32) // 2) - ((word1&8)*4) - ((word2&4)*16) - ((word2&8)*16) - ((word2&32)*8) - ((word2&1)*512) - ((word1&64)*16) - ((word1&16)*128) - ((word1&4)*1024) - ((word1&1)*8192)
	
			# event data

			ix=value
			iy=value

			# get word ident values
			
			TrgX = 1 - ((word&0x80)>>7)
			TrgY = 1 - ((word&0x200)>>9)
			TrgZ = (0b1 - TrgX)&(0b1-TrgY)

			TrgH = 1 - ((word&0x800000)>>23)
			TrgP = (word&0x400000)>>22

			# if(TrgX):
			# 	print("X");
			# if(TrgY):
			# 	print("Y");
			# if(TrgZ):
			# 	print("Z");
			# if(TrgH):
			# 	print("H");
			# if(TrgP):
			# 	print("P");
			# print("dtime = ", dtime, " itag= ", itime, " TimeStep = ", TimeStep)
			# print("itime = ", itime)

			# input('')
			
			# X word
			if(TrgX):
				# X pair 
				if(TrgP):
					pair = 1
					GotB = 1
				else:
					pair = 0
					GotA = 1
					GotB = 0
					GotXA = 0
					GotXB = 0
					GotYA = 0
					GotYB = 0

			if(TrgX):
				if(TrgH==0):
					ix1 = ix / 16
					GotXA = 1
				else:
					ix2 = ix / 16
					GotXB = 1



			if(TrgY):
				if(TrgH==0):
					iy1 = iy / 16
					GotYA = 1
				else:
					iy2 = 1024 - (iy / 16)
					GotYB = 1


			# Spit out the word as a single LOR

			# print("Bin status: GotA %i \t GotB %i \t GotXA %i \t GotXA %i \t GotYA %i \t GotYB %i \n" % (GotA, GotB, GotXA, GotXB, GotYA, GotYB))

			if (GotA + GotB + GotXA + GotXB + GotYA + GotYB) == 6:
				# print("Got an LOR!")

				GotXA = 0
				GotYA = 0
				GotXB = 0
				GotYB = 0

				GotA  = 0
				GotB  = 0
				nevent = nevent + 1

				ix2 = ix2 * (Gx2 / 1000)
				ix1 = ix1 * (Gx1 / 1000)
 
  	 	 	 	iy1 = iy1 * (Gy1 / 1000)
				iy2 = iy2 * (Gy2 / 1000)

				f_out.write("%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\n" % (itime, ix1, iy1, -10, ix2, iy2, separation+10))
				LORs.append([itime, ix1, ix2, -10, ix2, iy2, separation+10])
				# print("%i\t%5.3f\t%3.2f\t%3.2f\t%3.2f\t%3.2f\t%3.2f\n" % (itime, ix1, iy1, -10, ix2, iy2, separation+10))
				# input('')


			# word = f.read(4)

				# Write to a CSV

		f_out.close()

		print("Got %i LORs and wrote to CSV in %.3f seconds" % (nevent, time.process_time() - t0))
		print("File saved as %s" % saveas)

		return LORs