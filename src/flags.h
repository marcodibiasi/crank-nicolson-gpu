#ifndef FLAGS_H
#define FLAGS_H

#define VPRINTF(flags, ...) \
    do { if ((flags)->verbose) printf(__VA_ARGS__); } while(0)

typedef struct {
    int show_energy;
    int profile;
    char* profile_path;
    int verbose;
    int progress;
    int delta_save;
} Flags;

#endif
