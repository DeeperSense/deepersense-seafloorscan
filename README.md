# w-s3Tseg
Weakly Supervised Semantic Segmentation of Side-Scan Sonar Data

## How to run:
./train-val.sh

# connect to server
ssh -p 2221 student4@falcon.infoblitz.net

# run
cd student4/code/w-s3Tseg/
tmux new -s run_4_july_1
./_run/train.sh

# copy code to server
scp -r -P 2221 /home/alamdar11/SSS/w-s3Tseg student4@falcon.infoblitz.net:~/student4/code

# reattach
tmux attach -t run_5_july_1


# copy server to local
scp -r -P 2221 student4@falcon.infoblitz.net:~/scratch/w-s3Tseg/out/ISIM/sima_tiny_6-5_200_start10_end150_freq20_2 /home/alamdar11/SSS/outputs
