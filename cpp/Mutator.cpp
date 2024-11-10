//
// Created by bryce on 12/10/24.
//

#include "Mutator.h"

#include <bitset>
#include <iostream>

std::mt19937 Mutator::gen(std::random_device{}());
std::uniform_real_distribution<> Mutator::rng(0.0, 1.0);

void Mutator::mutateProgram(Program &program) {
    if (rng(gen) < Parameters::ADD_INSTRUCTION_PROBABILITY) {
        if (program.instructions.size() < Parameters::MAX_PROGRAM_LENGTH) {
            program.addRandomInstruction();
        }
    }
    if (rng(gen) < Parameters::DELETE_INSTRUCTION_PROBABILITY) {
        if (program.instructions.size() > 1) {
            std::uniform_int_distribution<> distrib(0, program.instructions.size() - 1);
            int index = distrib(gen);
            program.instructions.erase(program.instructions.begin() + index);
        }
    }
    if (rng(gen) < Parameters::SWAP_INSTRUCTION_PROBABILITY) {
        if (program.instructions.size() > 1) {
            std::uniform_int_distribution<> distrib(0, program.instructions.size() - 1);
            int idx1 = distrib(gen);
            int idx2 = distrib(gen);

            while (idx2 == idx1) {
                idx2 = distrib(gen);
            }

            std::swap(program.instructions[idx1], program.instructions[idx2]);
        }
    }

    if (rng(gen) < Parameters::SWAP_INSTRUCTION_PROBABILITY) {
        std::uniform_int_distribution<> distrib(0, program.instructions.size() - 1);
        int idx = distrib(gen);
        mutateInstruction(program.instructions[idx]);
    }
}

void Mutator::mutateInstruction(uint32_t &instruction) {
    const int INSTRUCTION_NUM_PARTS = 4;
    std::uniform_int_distribution<> distrib(0, INSTRUCTION_NUM_PARTS - 1);
    int index = distrib(gen);

    // Mutate the mode bit
    if (index == 0) {
        // Flips the 14th bit (mode bit)
        instruction ^= (MODE_MASK << MODE_SHIFT);

        // If the mode bit is 0, then we're using internal registers.
        // Make sure the source register is within the bounds.
        if (((instruction >> MODE_SHIFT) & MODE_MASK) == 0) {
            int src = (instruction) & SRC_MASK;
            src = src % Parameters::NUM_REGISTERS;
            instruction &= ~(SRC_MASK); // Clear the current source register
            instruction |= src; // Set the source register to the new clipped value
        }
    }
    // Mutate the op code
    else if (index == 1) {
        int opCode = (instruction >> OPCODE_SHIFT & OPCODE_MASK);

        std::uniform_int_distribution<> opDistrib(0, Parameters::NUM_OP_CODES - 1);
        int newOpCode = opDistrib(gen);

        if (newOpCode == opCode) {
            newOpCode = opDistrib(gen);
        }

        // Clear the current op code bits
        instruction &= ~(OPCODE_MASK << OPCODE_SHIFT);

        // Set the new op code
        instruction |= (newOpCode << OPCODE_SHIFT);
    }
    // Mutate the source register
    else if (index == 2) {
        int src = (instruction & SRC_MASK);

        int newSrc;
        // We're addressing internal registers (mode bit 0)
        if (((instruction >> MODE_SHIFT) & MODE_MASK) == 0) {
            std::uniform_int_distribution<> registerDistrib(0, Parameters::NUM_REGISTERS - 1);
            newSrc = registerDistrib(gen);

            if (newSrc == src) {
                newSrc = registerDistrib(gen);
            }
        }
        // We're addressing features (mode bit 1)
        else {
            std::uniform_int_distribution<> featureDistrib(0, Parameters::NUM_FEATURES - 1);
            newSrc = featureDistrib(gen);

            while (newSrc == src) {
                newSrc = featureDistrib(gen);
            }
        }

        instruction &= ~(SRC_MASK);
        instruction |= newSrc;
    }
    // Mutate the destination register
    else if (index == 3) {
        int dest = (instruction >> DEST_SHIFT & DEST_MASK);

        std::uniform_int_distribution<> registerDistrib(0, Parameters::NUM_REGISTERS - 1);
        int newDest = registerDistrib(gen);

        while (newDest == dest) {
            newDest = registerDistrib(gen);
        }

        instruction &= ~(DEST_MASK << DEST_SHIFT);
        instruction |= (newDest << DEST_SHIFT);
    }
    else
    {
        std::cerr << "Unknown mutation" << std::endl;
    }
}
