#include <stdio.h>
#include <stdlib.h>
// #include <stdint.h>
// #include <float.h>
// #include <math.h>
#include <iostream>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"

using namespace std;


int main(int argc, char *argv[]) {
    if(argc != 3)
    {
        cout << "Usage: ./kmeans datasets/dataset.arff k" << endl;
        exit(0);
    }

    int k = strtol(argv[2], NULL, 10);
}
