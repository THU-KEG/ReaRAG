# !/bin/bash

mkdir -p src_data/data

# Step 1: build conversation
python -m src_data.1_build_conversations \
        > src_data/data/conversations_qwq.log 2> src_data/data/conversations_qwq.err

# Step 2: clean conversation
python -m src_data.2_clean_conversations