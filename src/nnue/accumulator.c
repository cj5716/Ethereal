/******************************************************************************/
/*                                                                            */
/*    Ethereal is a UCI chess playing engine authored by Andrew Grant.        */
/*    <https://github.com/AndyGrant/Ethereal>     <andrew@grantnet.us>        */
/*                                                                            */
/*    Ethereal is free software: you can redistribute it and/or modify        */
/*    it under the terms of the GNU General Public License as published by    */
/*    the Free Software Foundation, either version 3 of the License, or       */
/*    (at your option) any later version.                                     */
/*                                                                            */
/*    Ethereal is distributed in the hope that it will be useful,             */
/*    but WITHOUT ANY WARRANTY; without even the implied warranty of          */
/*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           */
/*    GNU General Public License for more details.                            */
/*                                                                            */
/*    You should have received a copy of the GNU General Public License       */
/*    along with this program.  If not, see <http://www.gnu.org/licenses/>    */
/*                                                                            */
/******************************************************************************/

#include <immintrin.h>
#include <stdio.h>
#include <string.h>

#include "accumulator.h"
#include "nnue.h"
#include "types.h"

#include "../bitboards.h"
#include "../thread.h"
#include "../types.h"


static int sq64_to_sq32(int sq) {
    static const int Mirror[] = { 3, 2, 1, 0, 0, 1, 2, 3 };
    return ((sq >> 1) & ~0x3) + Mirror[sq & 0x7];
}

static int nnue_index(int piece, int relksq, int colour, int sq) {

    const int ptype   = pieceType(piece);
    const int pcolour = pieceColour(piece);
    const int relpsq  = relativeSquare(colour, sq);

    const int mksq = testBit(LEFT_FLANK, relksq) ? (relksq ^ 0x7) : relksq;
    const int mpsq = testBit(LEFT_FLANK, relksq) ? (relpsq ^ 0x7) : relpsq;

    return 640 * sq64_to_sq32(mksq) + (64 * (5 * (colour == pcolour) + ptype)) + mpsq;
}

int nnue_update_accumulator(NNUEAccumulator *accum, Board *board, int colour, int relksq) {

    NNUEAccumulator *accurate_accum = accum;
    int accums_seen = 0;

    // Search back through the tree to find an accurate accum
    while (true) {

        // We reached the root, without finding an accurate accum, so we have to refresh
        if (accurate_accum == &board->thread->nnue->stack[0])
            return FALSE;

        // A King move prevents the entire tree from being updated
        if (   accurate_accum->changes
            && accurate_accum->deltas[0].piece == makePiece(KING, colour))
            return FALSE;

        // Step back, since the root can't be accurate
        accums_seen++;
        accurate_accum = accurate_accum - 1;

        // We found it, so we can update the entire tree
        if (accurate_accum->accurate[colour])
            break;
    }

    for (int i = accums_seen - 1; i >= 0; i--) {
        NNUEAccumulator *current_accum = accum - i;

        // Determine the features that have changed, by looping through them
        for (NNUEDelta *x = &current_accum->deltas[0]; x < &current_accum->deltas[0] + current_accum->changes; x++) {

            // HalfKP does not concern with KxK relations
            if (pieceType(x->piece) == KING)
                continue;

            // Moving or placing a Piece to a Square
            if (x->to != SQUARE_NB)
                current_accum->add_list[current_accum->add++] = nnue_index(x->piece, relksq, colour, x->to);

            // Moving or deleting a Piece from a Square
            if (x->from != SQUARE_NB)
                current_accum->remove_list[current_accum->remove++] = nnue_index(x->piece, relksq, colour, x->from);
        }
    }

    vepi16 *inputs, *outputs, *weights, registers[NUM_REGS];
    for (int offset = 0; offset < KPSIZE; offset += NUM_REGS * vepi16_cnt) {
        inputs = (vepi16*) &accurate_accum->values[colour][offset];
        for (int i = 0; i < NUM_REGS; i++)
            registers[i] = inputs[i];

        for (int curr = accums_seen - 1; curr >= 0; curr--) {
            NNUEAccumulator *current_accum = accum - curr;

            for (int i = 0; i < current_accum->add; i++) {

                weights = (vepi16*) &in_weights[current_accum->add_list[i] * KPSIZE + offset];

                for (int j = 0; j < NUM_REGS; j++)
                    registers[j] = vepi16_add(registers[j], weights[j]);
            }

            for (int i = 0; i < current_accum->remove; i++) {

                weights = (vepi16*) &in_weights[current_accum->remove_list[i] * KPSIZE + offset];

                for (int j = 0; j < NUM_REGS; j++)
                    registers[j] = vepi16_sub(registers[j], weights[j]);
            }

            outputs = (vepi16*) &current_accum->values[colour][offset];
            for (int i = 0; i < NUM_REGS; i++)
                outputs[i] = registers[i];
        }
    }

    for (int i = accums_seen - 1; i >= 0; i--) {
        NNUEAccumulator *current_accum = accum - i;
        current_accum->add = 0;
        current_accum->remove = 0;
        current_accum->accurate[colour] = TRUE;
    }

    return TRUE;
}

void nnue_refresh_accumulator(NNUEEvaluator *nnue, NNUEAccumulator *accum, Board *board, int colour, int relsq) {

    vepi16 *outputs, *weights, registers[NUM_REGS];
    const int ksq = getlsb(board->pieces[KING] & board->colours[colour]);
    NNUEAccumulatorTableEntry *entry = &nnue->table[ksq];

    int set_indexes[32], set_count = 0;
    int unset_indexes[32], unset_count = 0;

    for (int c = WHITE; c <= BLACK; c++) {

        for (int pt = PAWN; pt <= QUEEN; pt++) {

            uint64_t pieces   = board->pieces[pt] & board->colours[c];
            uint64_t to_set   = pieces & ~entry->occupancy[colour][c][pt];
            uint64_t to_unset = entry->occupancy[colour][c][pt] & ~pieces;

            while (to_set)
                set_indexes[set_count++] = nnue_index(makePiece(pt, c), relsq, colour, poplsb(&to_set));

            while (to_unset)
                unset_indexes[unset_count++] = nnue_index(makePiece(pt, c), relsq, colour, poplsb(&to_unset));

            entry->occupancy[colour][c][pt] = pieces;
        }
    }

    for (int offset = 0; offset < KPSIZE; offset += NUM_REGS * vepi16_cnt) {

        outputs = (vepi16*) &entry->accumulator.values[colour][offset];

        for (int i = 0; i < NUM_REGS; i++)
            registers[i] = outputs[i];

        for (int i = 0; i < set_count; i++) {

            weights = (vepi16*) &in_weights[set_indexes[i] * KPSIZE + offset];

            for (int j = 0; j < NUM_REGS; j++)
                registers[j] = vepi16_add(registers[j], weights[j]);
        }

        for (int i = 0; i < unset_count; i++) {

            weights = (vepi16*) &in_weights[unset_indexes[i] * KPSIZE + offset];

            for (int j = 0; j < NUM_REGS; j++)
                registers[j] = vepi16_sub(registers[j], weights[j]);
        }

        for (int i = 0; i < NUM_REGS; i++)
            outputs[i] = registers[i];
    }

    memcpy(accum->values[colour], entry->accumulator.values[colour], sizeof(int16_t) * KPSIZE);
    accum->accurate[colour] = TRUE;
}