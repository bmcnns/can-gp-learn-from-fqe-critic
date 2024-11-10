//
// Created by bryce on 12/10/24.
//

#ifndef MUTATOR_H
#define MUTATOR_H

#include <random>
#include "Program.h"


class Mutator {
public:
    static void mutateProgram(Program& program);
private:
    static void mutateInstruction(uint32_t& instruction);
    static std::mt19937 gen;
    static std::uniform_real_distribution<> rng;
};

#endif //MUTATOR_H
