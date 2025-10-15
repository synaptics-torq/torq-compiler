// Copyright 2024 Synaptics Incorporated. All rights reserved.
// Created:  06/28/2024, Hongjie Guan
// Language: C99 with -fno-strict-aliasing

#ifndef TORQ_FILE_IO_H_
#define TORQ_FILE_IO_H_

#ifdef __cplusplus
extern "C" {
#endif


FILE  *torq_fopen(const char *filename, const char *mode, const char *fmt);
int    torq_fclose(FILE *stream);
size_t torq_fread(void *buffer, size_t size, size_t count, FILE *stream, const char *fmt);
size_t torq_fwrite(const void *buffer, size_t size, size_t count, FILE *stream, const char *fmt);


#ifdef __cplusplus
}
#endif

#endif
