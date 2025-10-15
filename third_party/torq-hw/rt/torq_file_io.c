// Copyright 2024 Synaptics Incorporated. All rights reserved.
// Created:  06/28/2024, Hongjie Guan
// Language: C99 with -fno-strict-aliasing

#include <stdio.h>
#include <string.h>
#include "torq_log.h"
#include "torq_file_io.h"

size_t _fread_hex(void *buffer, size_t size, size_t count, FILE *stream)
{
    int r, j, x;
    size_t i;
    unsigned char *p = buffer;
    if (!size || !count) return 0;
    for (i=0; i<count; i++, p+=size) {
        for (j=size-1; j>=0; j--) {
            while (0 < fscanf(stream, " %[#]%*[^\n]", (char *)&x)); //skip comments
            r = fscanf(stream, "%02x", &x);
            xdie_v(!r, "invalid character\n");
            if (r < 1) {
                xdie_v(j!=size-1, "incomplete record\n");
                return i;
            }
            p[j] = (unsigned char)x;
        } 
    }
    return i;
}

size_t _fwrite_hex(const void *buffer, size_t size, size_t count, FILE *stream)
{
    int r, j;
    size_t i;
    const unsigned char *p = buffer;
    if (!size || !count) return 0;
    for (i=0; i<count; i++, p+=size) {
        for (j=size-1; j>=0; j--) {
            r = fprintf(stream, "%02x", p[j]); xdie(r<1);
        } 
        r = fprintf(stream, "\n"); xdie(r<1);
    }
    return i;
}

size_t torq_fread(void *buffer, size_t size, size_t count, FILE *stream, const char *fmt)
{
    size_t r;
    if (!strcmp(fmt, "raw")) r = fread(buffer, size, count, stream); 
    else if (!strcmp(fmt, "hex")) r = _fread_hex(buffer, size, count, stream);
    else xdie_v(1, "unsupported file format: %s", fmt);
    return r;
}

size_t torq_fwrite(const void *buffer, size_t size, size_t count, FILE *stream, const char *fmt)
{
    size_t r;
    if (!strcmp(fmt, "raw")) r = fwrite(buffer, size, count, stream); 
    else if (!strcmp(fmt, "hex")) r = _fwrite_hex(buffer, size, count, stream);
    else xdie_v(1, "unsupported file format: %s", fmt);
    return r;
}

FILE *torq_fopen(const char *filename, const char *mode, const char *fmt)
{
    if (!strcmp(fmt, "raw")) { //special handling for Windows only
        int i;
        char m[8];
        for (i=0;(i<8-2)&&mode[i];i++) m[i] = mode[i];
        xdie_v(mode[i], "invalid mode: %s\n", mode);
	m[i++] = 'b';
	m[i] = 0;
        return fopen(filename, m);
    }
    return fopen(filename, mode);
}

int torq_fclose(FILE *stream)
{
    return fclose(stream);
}


#if 0
int main(int argc, char *argv[])
{
    FILE *fp = 0L;
    size_t n;
    char buf[256];
    fp = fopen("hex_input.txt", "r");
    n = _fread_hex(buf, 8, 32, fp);
    printf("fread_hex: n=%d\n", n);
    fclose(fp);
    fp = fopen("hex_output.txt", "w");
    n = _fwrite_hex(buf, 8, n, fp);
    printf("fwrite_hex: n=%d\n", n);
    fclose(fp);
    return 0;
}
#endif
