//
// Created by bryce on 12/10/24.
//

#ifndef PROGRAM_H
#define PROGRAM_H
#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include "Parameters.h"

#define OPCODE_MASK 0xF
#define OPCODE_SHIFT 27
#define SRC_MASK 0x3FFFFF
#define DEST_MASK 0x1F
#define DEST_SHIFT 22
#define MODE_MASK 0x1
#define MODE_SHIFT 31

class Program {
private:
    std::array<double, Parameters::NUM_REGISTERS> registers;
    int getRandomNumber(int min, int max);

public:
    std::vector<uint32_t> instructions;

    void addRandomInstruction();

    Program();

    void execute(const std::vector<double> &features);

    std::array<double, Parameters::NUM_REGISTERS> predict(const std::vector<double> &features);

    void displayInstructions() const;

    void displayRegisters() const;

    void displayCode() const;

    std::vector<uint32_t> __getstate__() const;

    void __setstate__(std::vector<uint32_t> state);
};

#endif //PROGRAM_H
