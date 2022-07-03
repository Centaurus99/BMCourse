#!/bin/bash

echo "" > summary.txt
for nhid in 100 300 1000
do
    for model in RNN LSTM GRU
    do
        for nlayer in 1 2
        do
            for dropout in 0.1 0.3
            do
                for lr in 0.5 0.1 0.05
                do
                    for epochs in 30
                    do
                        for batch_size in 8 16 32
                        do
                            echo "Running model: $model, nhid: $nhid, nlayer: $nlayer, dropout: $dropout, lr: $lr, epochs: $epochs, batch_size: $batch_size"
                            path=./models/${model}_${nhid}_${nlayer}_${dropout}_${lr}_${epochs}_${batch_size}
                            mkdir -p $path
                            python3 -u main.py --model $model --nhid $nhid --nlayer $nlayer --dropout $dropout --lr $lr --epochs $epochs --batch_size $batch_size --save "$path/model.pt" &> "$path/log.txt"
                            echo "Done: $path"
                            echo "model: $model, nhid: $nhid, nlayer: $nlayer, dropout: $dropout, lr: $lr, epochs: $epochs, batch_size: $batch_size" >> summary.txt
                            tail -n 3 "$path/log.txt" >> summary.txt
                            echo "" >> summary.txt
                        done
                    done
                done
            done
        done
    done
done