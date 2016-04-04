#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "types.h"
#include "transposition.h"

void init_transposition_table(TranspositionTable * table, int key_size){
	table->entries = malloc(sizeof(TranspositionEntry) * (1 << key_size));
	printf("Size of Table = %d MegaBytes\n",(sizeof(TranspositionEntry) * (1 << key_size))/(1024*1024));
	table->max_size = 1 << key_size;
	table->key_size = key_size;
	table->num_entries = 0;
	table->hits = 0;
	table->misses = 0;
	table->key_collisions = 0;
	
	int i;
	for (i = 0; i < table->max_size; i++){
		table->entries[i].depth = 0;
		table->entries[i].turn = 0;
		table->entries[i].type = 0;
		table->entries[i].value = 0;
		table->entries[i].best_move = 0;
		table->entries[i].hash = 0;
	}
}

TranspositionEntry * get_transposition_entry(TranspositionTable * table, uint64_t hash){
	TranspositionEntry * entry = &(table->entries[hash % table->max_size]);
	
	if (entry->hash != hash){
		if (entry->type != 0)
			table->key_collisions++;
		else
			table->misses++;
		return NULL;
	}
	
	table->hits++;	
	return entry;
}

void store_transposition_entry(TranspositionTable * table, int8_t depth, int8_t turn, int8_t type, int value, uint16_t best_move, uint64_t hash){
	TranspositionEntry * entry = &(table->entries[hash % table->max_size]);
	
	if (entry->type == 0){
		entry->depth = depth;
		entry->turn = turn;
		entry->type = type;
		entry->value = value;
		entry->best_move = best_move;
		entry->hash = hash;
		table->num_entries++;
		return;
	} 
	
	if (depth > entry->depth){
		entry->depth = depth;
		entry->turn = turn;
		entry->type = type;
		entry->value = value;
		entry->best_move = best_move;
		entry->hash = hash;
		return;
	}	
}