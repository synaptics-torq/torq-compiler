// Copyright 2024 Synaptics Incorporated. All rights reserved.
// Created:  12/10/2024, Hongjie Guan
// Language: C99 with -fno-strict-aliasing

#ifndef TORQ_OS_H_
#define TORQ_OS_H_


#ifdef _WIN32
    #include <direct.h>  // _mkdir
    #include <io.h>  // _access
    #ifdef mkdir
        #undef mkdir
    #endif
    #define mkdir(path, mode) _mkdir(path)
    #ifdef access
        #undef access
    #endif
    #define access _access
#else
    #include <sys/stat.h>  // mkdir
    #include <unistd.h>  // access
#endif


#ifdef __cplusplus
extern "C" {
#endif


static int os_exists(const char *path)
{
    return (access(path, 0) == 0);
}

static int os_mkdir(const char *path)
{
    int r = 0;
    if (!os_exists(path)) r = mkdir(path, 0755);
    return r;
}


#ifdef __cplusplus
}
#endif

#endif
