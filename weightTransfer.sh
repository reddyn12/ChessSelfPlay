#!/bin/bash
echo "Starting to transfer WEIGHTS"
# Rsync command to transfer file1 from remote server
# rsync -avz nr620@amarel.rutgers.edu:~/ChessSelfPlay/model_weights.pkl .

# Rsync command to transfer file2 from remote server
# rsync -avz nr620@amarel.rutgers.edu:~/ChessSelfPlay/model_weights_V2.pkl .

# Rsync command to transfer file3 from remote server
rsync -avz nr620@amarel.rutgers.edu:~/ChessSelfPlay/model_weights_PARALLEL.pkl .

echo "WEIGHTS transfer complete"