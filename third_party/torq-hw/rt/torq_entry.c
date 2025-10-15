// Copyright 2024 Synaptics Incorporated. All rights reserved.
// Created:  09/30/2024, Hongjie Guan
// Language: C99 with -fno-strict-aliasing

int torq_main(int argc, const char *argv[]);

#ifdef COSIM

static const char *argv[] = { "torq_rt_app", 0L, "-s", "." };

int sw_main(const char *arg) {
    int r;
    argv[1] = arg;
    r = torq_main(sizeof(argv)/sizeof(argv[0]), argv);
    return r<0?1:0;
}

#else

int main(int argc, const char *argv[]) {
    int r;
    r = torq_main(argc, argv);
    return r<0?1:0;
}

#endif
