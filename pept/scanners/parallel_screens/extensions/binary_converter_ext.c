/**
 * File   : binary_converter_ext.c
 * License: GNU v3.0
 * Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
 * Date   : 03.02.2021
 */


#include "binary_converter_ext.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>


#define HEADER_LEN 1000
#define ERROR_NOT_FOUND(word) fprintf(stderr, "Expected word `%s` not found!\n", (word));
#define CHECK_ALLOC(m)                              \
    {                                               \
        if ((m) == NULL)                            \
        {                                           \
            fclose(file);                           \
            perror("memory allocation failed");     \
            return NULL;                            \
        }                                           \
    }


typedef struct {
    int32_t separation;             // Screen separation
    int32_t angle;                  // Gantry angle
    char    header[HEADER_LEN];     // Data description
} Detector;


typedef struct {
    double          ix1, ix2, iy1, iy2;                 // LoR coordinates
    uint32_t        trg_x, trg_y, trg_z, trg_h, trg_p;  // Detector triggers
    uint_fast8_t    pair, got_first, got_second;        // Flags for finding LoR ends
    uint_fast8_t    got_xa, got_ya, got_xb, got_yb;     // Flags for finding LoR coordinates
} EventCache;


typedef struct {
    double itag;                    // Time tag in current word
    double itime;                   // Actual time in ms
    double itagold;                 // Time tag in last word
    double dtime;                   // Difference in time tags between current and last word

    double timestep;                // Calculated period of the acqusition board clock
} TimeCache;


FILE*           open_file(const char* filepath)
{
    FILE*       file;

    file = fopen(filepath, "rb");
    if (file == NULL)
        fprintf(stderr, "File `%s` not found!\n", filepath);

    return file;
}


uint_fast8_t    init_detector(FILE* file, Detector* detector)
{
    // `fread` returns the number of elements read; check it is not zero
    if (fread(&detector->separation, sizeof(detector->separation), 1, file) != 1)
    {
        ERROR_NOT_FOUND("screen separation");
        return 0;
    }

    if (fread(&detector->angle, sizeof(detector->angle), 1, file) != 1)
    {
        ERROR_NOT_FOUND("gantry angle");
        return 0;
    }

    if (fread(&detector->header, sizeof(char), HEADER_LEN, file) != HEADER_LEN)
    {
        ERROR_NOT_FOUND("header info");
        return 0;
    }

    return 1;
}


void            init_event_cache(EventCache* ec)
{
    ec->trg_x = 0;
    ec->trg_y = 0;
    ec->trg_z = 0;
    ec->trg_h = 0;
    ec->trg_p = 0;

    ec->pair = 0;
    ec->got_xa = 0;
    ec->got_ya = 0;
    ec->got_xb = 0;
    ec->got_yb = 0;
    ec->got_first = 0;
    ec->got_second = 0;
}


/**
 *  Bit twiddling magic, taken from the old ADAC list mode binary format decoder used at
 *  the University of Birmingham, developed by Dr. Tom Leadbeater.
 */
void            update_event_cache(EventCache* ec, const uint32_t word)
{
    uint32_t    word_cut;
    uint32_t    word_p1;
    uint32_t    word_p2;
    uint32_t    value;

    word_cut = word & 0xFFFFFFA0;
    word_p1 = (word_cut & 0X0000FF00) >> 8;
    word_p2 = (word_cut & 0X00FF0000) >> 16;

    value = (uint32_t)(
        16383 -
        ((word_p2 & 16) >> 4) -
        ((word_p2 & 64) >> 5) -
        ((word_p2 & 2) << 1) -
        ((word_p1 & 128) >> 4) -
        ((word_p1 & 32) >> 1) -
        ((word_p1 & 8) << 2) -
        ((word_p2 & 4) << 4) -
        ((word_p2 & 8) << 4) -
        ((word_p2 & 32) << 3) -
        ((word_p2 & 1) << 9) -
        ((word_p1 & 64) << 4) -
        ((word_p1 & 16) << 7) -
        ((word_p1 & 4) << 10) -
        ((word_p1 & 1) << 13)
    );

    value = value >> 4;

    ec->trg_x = 1 - ((word_cut & 0x80) >> 7);
    ec->trg_y = 1 - ((word_cut & 0x200) >> 9);
    ec->trg_z = 1 - ec->trg_x - ec->trg_z;

    ec->trg_h = 1 - ((word_cut & 0x800000) >> 23);
    ec->trg_p = (word_cut & 0x400000) >> 22;

    if (ec->trg_x == 1)
    {
        if (ec->trg_p == 1)
        {
            ec->pair = 1;
            ec->got_second = 1;
        }
        else
        {
            ec->pair = 0;
            ec->got_first = 1;
            ec->got_xa = 0;
            ec->got_ya = 0;
            ec->got_xb = 0;
            ec->got_yb = 0;
        }

        if (ec->trg_h == 1)
        {
            ec->ix2 = (double)value * 0.59;
            ec->got_xb = 1;
        }
        else
        {
            ec->ix1 = (double)value * 0.59;
            ec->got_xa = 1;
        }
    }

    if (ec->trg_y == 1)
    {
        if (ec->trg_h == 1)
        {
            ec->iy2 = (double)(1024 - value) * 0.59;
            ec->got_yb = 1;
        }
        else
        {
            ec->iy1 = (double)value * 0.59;
            ec->got_ya = 1;
        }
    }
}


uint_fast8_t    full_event_cache(const EventCache* ec)
{
    return (
        ec->got_xa &&
        ec->got_xb &&
        ec->got_ya &&
        ec->got_yb &&
        ec->got_first &&
        ec->got_second
    );
}


void            reset_event_cache(EventCache* ec)
{
    ec->trg_x = 0;
    ec->trg_y = 0;
    ec->trg_z = 0;
    ec->trg_h = 0;
    ec->trg_p = 0;

    ec->pair = 0;
    ec->got_xa = 0;
    ec->got_ya = 0;
    ec->got_xb = 0;
    ec->got_yb = 0;
    ec->got_first = 0;
    ec->got_second = 0;
}


/**
 * Calculate clock frequency of the detector timeboard; code contributed by Dawid M. Hampel.
 */
uint_fast8_t    calculate_time_cache(FILE* file, TimeCache* tc)
{
    uint32_t    word;

    // Linear fit parameters to compute timestep per tag
    double      s = 0;
    double      sx = 0;
    double      sy = 0;
    double      sxx = 0;
    double      sxy = 0;
    double      delta;

    int32_t     buftime = 0;        // Time of the whole buffer from CPU Time in ms
    int32_t     cputimestart;       // Initial PC time, usually 0 or very close to it

    // Initialise TimeCache
    tc->itag = 0;
    tc->itime = 0;
    tc->itagold = 0;
    tc->dtime = 0;

    // Read in the face flag, first recorded CPU timestamp, and first itag
    // If the word is the timing flag 0xFACEFACE, then the next word is the recorded CPU time
    // at the start of a new buffer
    if (!fread(&word, sizeof(word), 1, file) || word != 0xFACEFACE)
    {
        ERROR_NOT_FOUND("timing flag 0xFACEFACE after header");
        return 0;
    }
    else
    {
        if (!fread(&word, sizeof(word), 1, file))
        {
            ERROR_NOT_FOUND("CPU time start");
            return 0;
        }
        cputimestart = word;    // CPU time in ms

        if (!fread(&word, sizeof(word), 1, file))
        {
            ERROR_NOT_FOUND("expected a word");
            return 0;
        }
        tc->itag = ((word & 0xFF000000) >> 24);
    }

    // Read in all itags, CPUtimes, and use linear fit to get the period/timestep of each itag tick
    while (1)
    {
        if (!fread(&word, sizeof(word), 1, file))
            break;

        // Start of a new buffer
  		if (word == 0xFACEFACE)
        {
            if (!fread(&word, sizeof(word), 1, file))
                break;

            buftime = word - cputimestart ;

            // fit to linear
            s++;
            sx += (double)buftime;
            sy += tc->itime;
            sxx += (double)buftime * buftime;
            sxy += (double)buftime * tc->itime;

            continue;
  		}

  		tc->itagold = tc->itag;
  		tc->itag = ((word & 0xFF000000) >> 24);
  		tc->dtime = tc->itag - tc->itagold;

  		if (tc->dtime < 0)
            tc->dtime += 256;

  		if (tc->dtime < 130)
            tc->itime += tc->dtime;
    }

  	delta = s * sxx - sx * sx;
    tc->timestep = (s * sxy - sx * sy) / delta;     // gradient

    return 1;
}


void            reallocate_lors(double** lors, size_t lors_rows)
{
    double*     new_lors = (double*)realloc(*lors, sizeof(double) * lors_rows  * 7);

    if (new_lors == NULL)
    {
        perror("memory allocation failed");
        free(*lors);
    }
    else
        *lors = new_lors;
}


/** Binary converter for the ADAC Forte dual-head gamma camera native list mode data. Given the
 *  `filepath` to such a binary file (usually with extension ".da01"), this function converts the
 *  binary contents to the general line of response format `[time, x1, y1, z1, x2, y2, z2]`, where
 *  `z1 = 0` and `z2 = screen_separation`.
 *
 *  The LoRs are retured in a row-major *flattened* vector of `double`s. For example, if there are
 *  10 LoRs in a file, then the returned vector will have 10 rows * 7 columns = 70 elements.
 *
 *  The flattened array is returned and its ownership is given to the caller. The total number of
 *  elements in the returned vector is saved in the second function parameter, `lors_elements`. If
 *  no LoRs were read in, NULL is returned and `lors_elements` is set to zero.
 *
 *  Function parameters
 *  -------------------
 *  filepath: const char*
 *      A C string of characters containing the path to the binary file. The string's contents are
 *      not read in - it will only be used with the `fopen` function, so it can contain any
 *      characters allowed by the OS file system.
 *
 *  lors_elements: ssize_t*
 *      This "inout" parameter will be set to the number of elements in the returned vector.
 *
 *  Returns
 *  -------
 *  lors: double* or NULL
 *      A flattened 2D array of `double`s containing the time and coordinates of the first and
 *      second point defining a 3D line, respectively: [time, x1, y1, z1, x2, y2, z2]. The
 *      total number of elements (number_of_lors * 7) is saved in the `lors_elements` input
 *      parameter. If you need the number of LoRs, simply divide that by seven (the number of
 *      columns). IMPORTANT: if no LoRs were read in, NULL is returned and `lors_elements` is
 *      set to zero.
 *
 *  Notes
 *  -----
 *  This array is heap-allocated and its ownership is returned to the caller - do not forget to
 *  deallocate it (or encapsulate it in a reference-counted container like a Python object)!
 *
 *  Errors
 *  ------
 *  If OS memory allocation fails, or the file at `filepath` is corrupted or inexistent, NULL is
 *  returned and `*lors_elements` is set to zero.
 */
double*         read_adac_binary(const char* filepath, ssize_t* lors_elements)
{
    FILE*       file;                   // Opened file descriptor
    Detector    detector;               // Detector data from beginning of file
    EventCache  event_cache;            // Coincidence event cache
    TimeCache   time_cache;             // Coincidence timing cache

    double*     lors;                   // Flattened (N, 7) array of LoR coordinates
    double*     current_lor;            // Pointer to current LoR line in `lors`
    size_t      lors_rows;              // Number of rows N in the flattened array
    size_t      lors_idx;               // Current `lors` **row** index

    uint32_t    word;

    // Initialise the output number of LoRs to zero in case NULL is returned on error
    *lors_elements = 0;

    // Initialise file handler and read in header data into a `Detector` struct
    file = open_file(filepath);
    if (file == NULL || !init_detector(file, &detector))
        return NULL;

    // Go through file and calculate the clock frequency
    if (!calculate_time_cache(file, &time_cache))
    {
        fclose(file);
        return NULL;
    }

    // Re-initialise file start and tags
    rewind(file);

    time_cache.itime = 0;
    time_cache.itag = 0;
    time_cache.itagold = 0;

    init_detector(file, &detector);

    // Initialise event cache
    init_event_cache(&event_cache);

    // Initialise the LoRs flattened array with seven columns [time, x1, y1, z1, x2, y2, z2]
    lors_rows = 40000;
    lors = (double*)malloc(sizeof(double) * lors_rows * 7);
    CHECK_ALLOC(lors);

    lors_idx = 0;

    while (1)
    {
        // If a full LoR was read in, save it in `lors`
        if (full_event_cache(&event_cache))
        {
            // If `lors` is full, reallocate it to a larger size
            if (lors_idx >= lors_rows)
            {
                lors_rows += lors_rows / 2;
                reallocate_lors(&lors, lors_rows);
                CHECK_ALLOC(lors);
            }

            // Move pointer to current LoR index
            current_lor = lors + lors_idx * 7;

            // Calculated time of the last of the 6 words that make the LOR
            current_lor[0] = time_cache.itime;

            current_lor[1] = (double)(event_cache.ix1);
            current_lor[2] = (double)(event_cache.iy1);
            current_lor[3] = -10.0;

            current_lor[4] = (double)(event_cache.ix2);
            current_lor[5] = (double)(event_cache.iy2);
            current_lor[6] = (double)(detector.separation) + 10.0;

            lors_idx += 1;

            reset_event_cache(&event_cache);
        }

        // Read word after EventCache check to avoid losing the last LoR...
        if (!fread(&word, sizeof(word), 1, file))
            break;

        // If a new buffer is encountered, skip the CPU time tag
        if (word == 0xFACEFACE)
        {
            if (!fread(&word, sizeof(word), 1, file))
                break;

            continue;
        }

        time_cache.itagold = time_cache.itag;
        time_cache.itag = ((word & 0xFF000000) >> 24);
        time_cache.dtime = time_cache.itag - time_cache.itagold;

        if (time_cache.dtime < 0)
            time_cache.dtime += 256;

        if (time_cache.dtime < 130)
            time_cache.itime += time_cache.dtime / time_cache.timestep;

        update_event_cache(&event_cache, word);
    }

    // If no LoRs were read in, free allocated data and return NULL
    if (lors_idx == 0)
    {
        free(lors);
        fclose(file);
        return NULL;
    }

    // Resize `lors` array to truncate unwritten elements
    reallocate_lors(&lors, lors_idx);
    CHECK_ALLOC(lors);
    *lors_elements = (ssize_t)(lors_idx * 7);

    // Close stream to the binary file
    fclose(file);

    return lors;
}


/* Uncomment for a standalone executable binary reader
//
// Main defined to create standalone executable, called with the filepath as input, e.g.:
// $> clang converter.c
// $> ./a.out <filepath>
int main(int argc, char **argv)
{
    double      *lors;          // Point to flattened array of LoRs
    ssize_t     num_elements;   // Number of elements in `lors` array
    double      *cl;            // Pointer to the current LoR when printing
    if (argc != 2)
    {
        fprintf(stderr, "Incorrect input arguments. Usage:\n%s <binary file path>\n", argv[0]);
        return EXIT_FAILURE;
    }
    // This function allocates the memory needed by lors and sets the second argument to the
    // number of values in it
    lors = read_adac_binary(argv[1], &num_elements);
    // Print LoR coordinates as [time, x1, y1, z1, x2, y2, z2] to stdout - that can be redirected
    // to a file if needed
    if (num_elements > 0)
        printf("time x1 y1 z1 x2 y2 z2");
    for (ssize_t i = 0; i < num_elements / 7; ++i)
    {
        cl = lors + i * 7;
        printf("%f %f %f %f %f %f %f\n", cl[0], cl[1], cl[2], cl[3], cl[4], cl[5], cl[6]);
    }
    free(lors);
    return EXIT_SUCCESS;
}
*/
