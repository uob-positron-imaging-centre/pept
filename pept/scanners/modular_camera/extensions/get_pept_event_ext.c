
/**
 *   pept is a Python library that unifies Positron Emission Particle
 *   Tracking (PEPT) research, including tracking, simulation, data analysis
 *   and visualisation tools
 *
 *   Copyright (C) 2019 Andrei Leonard Nicusan
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

/**
 * File              : get_pept_event_ext.c
 * License           : License: GNU v3.0
 * Author            : Sam Manger
 * Date              : 01.07.2019
 */


#include "get_pept_event_ext.h"


unsigned short int bitrev(unsigned short int i1, int length)
{
    unsigned short int j,k,dum,bit1,bit2,out;
    out=0;
    for(j=0;(int)j<length;j++){
        k=length-1-j;
        dum=1<<j;
        bit1=dum&i1;
        if(k>j)
            bit2=bit1<<(k-j);
        else
            bit2=bit1>>(j-k);
        out=out|bit2;
    }
    return out;
}



void get_pept_event_ext(double* result, unsigned int word, int itag, int itime)
{  
    int MPnum[112] = 
    {   0,5,  0,6,  0,7,  0,8,  0,9,  0,10, 0,11, 
        1,6,  1,7,  1,8,  1,9,  1,10, 1,11, 1,12,
        2,7,  2,8,  2,9,  2,10, 2,11, 2,12, 2,13,
        3,8,  3,9,  3,10, 3,11, 3,12, 3,13, 3,14,
        4,9,  4,10, 4,11, 4,12, 4,13, 4,14, 4,15,
        5,10, 5,11, 5,12, 5,13, 5,14, 5,15,
        6,11, 6,12, 6,13, 6,14, 6,15, 
        7,12, 7,13, 7,14, 7,15, 
        8,13, 8,14, 8,15,
        9,14, 9,15,
        10,15                       
    };

    unsigned short int short1,short2,word1,word2;
    int dtime, modpair,BoardAddr,Bucket[2],Block[2],Seg[2],Plane[2],itagold;

    short1 = (word&0xffff);
    word1=bitrev(short1,16);   //reverse bit ordering for PJ card
    short2 = (word&0xffff0000)>>16;
    word2=bitrev(short2,16);

    itagold=itag;
    itag=word2&0x0f;
    dtime=itag-itagold;
    if(dtime<0)dtime+=16;
    itime+=dtime*2;

    modpair = word1&0x3f;

    Bucket[0] = MPnum[2*modpair-2];   //check whether starts from 0 or 1
    Bucket[1] = MPnum[2*modpair-1];

    Block[0] = (word2&0x180)>>7;
    Block[1] = (word2&0x6000)>>13;

    Seg[0] = (word2&0x1c00)>>10;
    Seg[1] = (word2&0x70)>>4;

    BoardAddr  = (word1 & 0xc000)>>14;

    if(BoardAddr==1||BoardAddr==2)
        BoardAddr=3-BoardAddr;

    if(BoardAddr == 0);
    else if(BoardAddr == 1) 
    {   

        Bucket[1] += 16;
    }
    else if(BoardAddr == 2)
    {

        Bucket[0] += 16;
    }
    else if(BoardAddr == 3) 
    {
        Bucket[0] += 16;
        Bucket[1] += 16;
    }

    Plane[0] = (word1&0x300)>>8;
    Plane[1] = (word1&0x1800)>>11;

    int it;

    if (modpair>28) // should be 56 but for modular camera we cop out at 28
    {
        for (it=0; it<12; it++)
        {
            result[it] = 0;   
        }
    }
    else
    {
        result[0] = word;
        result[1] = itag;
        result[2] = itime;
        result[3] = modpair;
        result[4] = Bucket[0];
        result[5] = Bucket[1];
        result[6] = Block[0];
        result[7] = Block[1];
        result[8] = Seg[0];
        result[9] = Seg[1];
        result[10] = Plane[0];
        result[11] = Plane[1];
    }
}


void get_pept_LOR_ext(double* result, unsigned int word, int itag, int itime)
{  
    int MPnum[112] = 
    {   0,5,  0,6,  0,7,  0,8,  0,9,  0,10, 0,11, 
        1,6,  1,7,  1,8,  1,9,  1,10, 1,11, 1,12,
        2,7,  2,8,  2,9,  2,10, 2,11, 2,12, 2,13,
        3,8,  3,9,  3,10, 3,11, 3,12, 3,13, 3,14,
        4,9,  4,10, 4,11, 4,12, 4,13, 4,14, 4,15,
        5,10, 5,11, 5,12, 5,13, 5,14, 5,15,
        6,11, 6,12, 6,13, 6,14, 6,15, 
        7,12, 7,13, 7,14, 7,15, 
        8,13, 8,14, 8,15,
        9,14, 9,15,
        10,15                       
    };

    unsigned short int short1,short2,word1,word2;

    int dtime, modpair,BoardAddr,Bucket[2],Block[2],Seg[2],Plane[2],itagold;

    float x[2],y[2],z[2];

    float modHeight=95.;
    float segHeight=13.5;
    float segWidth=6.25;
    float blockWidth=50;
    float blockSep=41;
    float modSep=250;

    short1 = (word&0xffff);
    word1=bitrev(short1,16);   //reverse bit ordering for PJ card
    short2 = (word&0xffff0000)>>16;
    word2=bitrev(short2,16);

    itagold=itag;
    itag=word2&0x0f;
    dtime=itag-itagold;
    if(dtime<0)dtime+=16;
    itime+=dtime*2;

    modpair = word1&0x3f;

    Bucket[0] = MPnum[2*modpair-2];   //check whether starts from 0 or 1
    Bucket[1] = MPnum[2*modpair-1];

    Block[0] = (word2&0x180)>>7;
    Block[1] = (word2&0x6000)>>13;

    Seg[0] = (word2&0x1c00)>>10;
    Seg[1] = (word2&0x70)>>4;

    BoardAddr  = (word1 & 0xc000)>>14;

    if(BoardAddr==1||BoardAddr==2)
        BoardAddr=3-BoardAddr;

    if(BoardAddr == 0);
    else if(BoardAddr == 1) 
    {   

        Bucket[1] += 16;
    }
    else if(BoardAddr == 2)
    {

        Bucket[0] += 16;
    }
    else if(BoardAddr == 3) 
    {
        Bucket[0] += 16;
        Bucket[1] += 16;
    }

    Plane[0] = (word1&0x300)>>8;
    Plane[1] = (word1&0x1800)>>11;

    int detector;

    for (detector=0; detector<2; detector++)
    {

        if(0 <= Bucket[detector] && Bucket[detector] <= 3){
            x[detector] = modSep;
            y[detector] = ((3-Bucket[detector])*modHeight) + (Plane[detector] * segHeight);
            z[detector] = ((2-Block[detector])*(blockWidth+blockSep) - (blockSep/2) - (segWidth/2) - ((7-Seg[detector])*segWidth));
        }
        else if (8 <= Bucket[detector] && Bucket[detector] <= 11){
            x[detector] = -modSep;
            y[detector] = ((11-Bucket[detector])*modHeight) + (Plane[detector] * segHeight);
            z[detector] = -((2-Block[detector])*(blockWidth+blockSep) - (blockSep/2) - (segWidth/2) - ((7-Seg[detector])*segWidth));
        }
        else if (16 <= Bucket[detector] && Bucket[detector] <= 19){
            x[detector] = ((2-Block[detector])*(blockWidth+blockSep) - (blockSep/2) - (segWidth/2) - ((7-Seg[detector])*segWidth));
            y[detector] = ((19-Bucket[detector])*modHeight) + (Plane[detector] * segHeight);
            z[detector] = -modSep;
        }
        else if (24 <= Bucket[detector] && Bucket[detector] <= 27){
            x[detector] = -((2-Block[detector])*(blockWidth+blockSep) - (blockSep/2) - (segWidth/2) - ((7-Seg[detector])*segWidth));
            y[detector] = ((27-Bucket[detector])*modHeight) + (Plane[detector] * segHeight);
            z[detector] = +modSep;
        }
        else
        {
            continue;
        }

    }

    int it;

    if (modpair>28) // should be 56 but for modular camera we cop out at 28
    {
        for (it=0; it<8; it++)
        {
            result[it] = 0;   
        }
    }
    else
    {
        result[0] = itag;
        result[1] = itime;
        result[2] = x[0];
        result[3] = y[0];
        result[4] = z[0];
        result[5] = x[1];
        result[6] = y[1];
        result[7] = z[1];
    }
}
