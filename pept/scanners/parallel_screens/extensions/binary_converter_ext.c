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




void            error_not_found(const char *word_tried)
{
    fprintf(stderr, "Expected word `%s` not found!\n", word_tried);
    exit(EXIT_FAILURE);
}


FILE*           open_file(const char *filepath)
{
    FILE        *file;

    file = fopen(filepath, "r");
    if (file == NULL)
    {
        fprintf(stderr, "File `%s` not found!\n", filepath);
        exit(EXIT_FAILURE);
    }

    return file;
}


Detector        init_detector(FILE *file)
{
    Detector    detector;

    // `fread` returns the number of elements read; check it is not zero
    if (fread(&detector.separation, sizeof(detector.separation), 1, file) != 1)
        error_not_found("screen separation");

    if (fread(&detector.angle, sizeof(detector.angle), 1, file) != 1)
        error_not_found("gantry angle");

    if (fread(&detector.header, sizeof(char), HEADER_LEN, file) != HEADER_LEN)
        error_not_found("header info");

    return detector;
}


EventCache      init_event_cache()
{
    EventCache  ec;

    ec.trg_x = 0;
    ec.trg_y = 0;
    ec.trg_z = 0;
    ec.trg_h = 0;
    ec.trg_p = 0;

    ec.pair =          0;
    ec.got_xa =        0;
    ec.got_ya =        0;
    ec.got_xb =        0;
    ec.got_yb =        0;
    ec.got_first =     0;
    ec.got_second =    0;

    return ec;
}


/**
 *  Bit twiddling magic, taken from the old ADAC list mode binary format decoder used at
 *  the University of Birmingham
 */
void            update_event_cache(EventCache *ec, const uint32_t word)
{
    uint32_t    word_cut;
    uint32_t    word_p1;
    uint32_t    word_p2;
    double      value;

    word_cut = word & 0xFFFFFFA0;
    word_p1 = (word_cut & 0X0000FF00) >> 8;
    word_p2 = (word_cut & 0X00FF0000) >> 16;

    value = (double)(
        16383 -
        (uint32_t)((word_p2 & 16) / 16) -       // Fix clang-tidy's bugprone-integer-division
        (uint32_t)((word_p2 & 64) / 32) -       // warning for dividing integers in float context
        (word_p2 & 2) * 2 -
        (uint32_t)((word_p1 & 128) / 16) -
        (uint32_t)((word_p1 & 32) / 2) -
        (word_p1 & 8) * 4 -
        (word_p2 & 4) * 16 -
        (word_p2 & 8) * 16 -
        (word_p2 & 32) * 8 -
        (word_p2 & 1) * 512 -
        (word_p1 & 64) * 16 -
        (word_p1 & 16) * 128 -
        (word_p1 & 4) * 1024 -
        (word_p1 & 1) * 8192
    );

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
            ec->ix2 = value / 16;
            ec->got_xb = 1;
        }
        else
        {
            ec->ix1 = value / 16;
            ec->got_xa = 1;
        }
    }

    if (ec->trg_y == 1)
    {
        if (ec->trg_h == 1)
        {
            ec->iy2 = 1024.0 - (value / 16);
            ec->got_yb = 1;
        }
        else
        {
            ec->iy1 = value / 16;
            ec->got_ya = 1;
        }
    }
}


uint_fast8_t    full_event_cache(const EventCache *ec)
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


void            reset_event_cache(EventCache *ec)
{
    ec->trg_x = 0;
    ec->trg_y = 0;
    ec->trg_z = 0;
    ec->trg_h = 0;
    ec->trg_p = 0;

    ec->pair =          0;
    ec->got_xa =        0;
    ec->got_ya =        0;
    ec->got_xb =        0;
    ec->got_yb =        0;
    ec->got_first =     0;
    ec->got_second =    0;
}


double          set_lors_times(double *lors, const size_t lors_idx_prev, const size_t lors_idx,
                               const uint32_t time_prev, const uint32_t time)
{
    size_t      idx;                    // LoR iterator
    double      time_increment;         // Linear time increment between two LoRs in this buffer

    time_increment = (double)(time - time_prev) / (lors_idx - lors_idx_prev);

    for (idx = lors_idx_prev; idx < lors_idx; ++idx)
        (lors + idx * 7)[0] = time_prev + (idx - lors_idx_prev) * time_increment;

    return time_increment;
}


void            reallocate_lors(double **lors, size_t *lors_rows)
{
    *lors = (double*)realloc(
        *lors,
        sizeof(double) * (*lors_rows + *lors_rows / 2) * 7
    );

    *lors_rows += *lors_rows / 2;
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
 *  lors: double*
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
 *  Undefined Behaviour
 *  -------------------
 *  This function assumes you have enough memory to load all the LoRs in the file into the RAM.
 *  If the OS cannot allocate enough memory for the LoRs being read, calls to `malloc` will fail.
 *
 *  Otherwise, this function should be safe to run, even with inexistent or corrupted files.
 */
double*         read_adac_binary(const char *filepath, ssize_t *lors_elements)
{
    FILE        *file;                  // Opened file descriptor
    Detector    detector;               // Detector data from beginning of file
    EventCache  event_cache;            // Coincidence event cache

    double      *lors;                  // Flattened (N, 7) array of LoR coordinates
    double      *current_lor;           // Pointer to current LoR line in `lors`
    size_t      lors_rows;              // Number of rows N in the flattened array
    size_t      lors_idx;               // Current `lors` **row** index
    size_t      lors_idx_prev;          // Previous buffer's `lors` row index
    size_t      idx;                    // Iterator

    uint32_t    time_start, time, time_prev;
    uint32_t    word;
    double      time_increment;         // Time increment between LoRs per buffer

    // Initialise file handler and read in header data into a `Detector` struct
    file = open_file(filepath);
    detector = init_detector(file);

    // Initialise event cache
    event_cache = init_event_cache();

    // Initialise the LoRs flattened array with seven columns [time, x1, y1, z1, x2, y2, z2]
    lors_rows = 40000;
    lors = (double*)malloc(sizeof(double) * lors_rows * 7);

    lors_idx = 0;
    lors_idx_prev = 0;

    time_increment = 0;

    // Read in the first recorded CPU timestamp
    if (fread(&word, sizeof(word), 1, file) != 1 || word != 0xFACEFACE)
        error_not_found("timing flag 0xFACEFACE after header");
    else
    {
        if (fread(&word, sizeof(word), 1, file) != 1)
            error_not_found("CPU time start");

        time_start = word;
        time = time_start;
        time_prev = time_start;
    }

    while (fread(&word, sizeof(word), 1, file))
    {
        // If a full LoR was read in, save it in `lors`
        if (full_event_cache(&event_cache))
        {
            // If `lors` is full, reallocate it to a larger size
            if (lors_idx >= lors_rows)
                reallocate_lors(&lors, &lors_rows);

            // Move pointer to current LoR index
            current_lor = lors + lors_idx * 7;

            // Time will be set when the timestamp of the next buffer is found
            current_lor[1] = (double)(event_cache.ix1);
            current_lor[2] = (double)(event_cache.iy1);
            current_lor[3] = 0.0;

            current_lor[4] = (double)(event_cache.ix2);
            current_lor[5] = (double)(event_cache.iy2);
            current_lor[6] = (double)(detector.separation);

            lors_idx += 1;

            reset_event_cache(&event_cache);
        }

        // If the word is the timing flag 0xFACEFACE, then the next word is the recorded CPU time
        // at the start of a new buffer
        if (word == 0xFACEFACE)
        {
            // If the buffer is empty, stop the loop
            if (fread(&word, sizeof(word), 1, file) != 1)
                break;

            time = word;

            // Set the times of the LoRs in the previous buffer as equally-spaced timestamps
            // between the previous buffer's time and this new buffer's time
            time_increment = set_lors_times(
                lors,
                lors_idx_prev,
                lors_idx,
                time_prev - time_start,
                time - time_start
            );

            lors_idx_prev = lors_idx;
            time_prev = time;

            continue;
        }

        update_event_cache(&event_cache, word);
    }

    // If loop ended without a new buffer, set the last LoRs' times as a continuation of the
    // time increments in the previous buffer
    if (lors_idx_prev < lors_idx)
        for (idx = lors_idx_prev; idx < lors_idx; ++idx)
            (lors + idx * 7)[0] = time - time_start + (idx - lors_idx_prev) * time_increment;

    // Close stream to the binary file
    fclose(file);

    // If no LoRs were read in, free allocated data and return NULL
    if (lors_idx == 0)
    {
        free(lors);
        *lors_elements = 0;

        return NULL;
    }

    // Resize `lors` array to truncate unwritten elements
    lors = (double*)realloc(lors, sizeof(double) * lors_idx * 7);
    *lors_elements = (ssize_t)(lors_idx * 7);

    return lors;
}


/** Uncomment for a standalone executable binary reader
 *
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
