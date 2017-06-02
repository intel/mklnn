#ifndef RANDOM_H
#define RANDOM_H
typedef unsigned long RNG;

inline RNG RNGInit(long seed)
{
    RNG rng = seed ? (unsigned long)seed : (unsigned long)(long)-1;
    return rng;
}

inline unsigned int RandInt( RNG* rng )
{
    unsigned long temp = *rng;
    temp = (unsigned long)(unsigned)temp*1554115554 + (temp >> 32);
    *rng = temp;
    return (unsigned long)temp;
}
#endif
