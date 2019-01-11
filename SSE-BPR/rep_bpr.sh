i=0.01
p1=1.0
p2=1.0
for j in 1 5 10 15 20 25 30 35 40 
do
    echo $i
    echo $j
    echo $p1 $p2
    ./bin/bpr --train_dataset=./ml1m-50-oc/train.txt --test_dataset=./ml1m-50-oc/test.txt --user_factors=W.txt --item_factors=H.txt --nfactors=100 --nepochs=$j --user_lambda=$i --item_lambda=$i --init_learning_rate=0.1 --num_negative_samples=3 -nthreads=16 --user_threshold=$p1 --item_threshold=$p2
    mv W.txt W_bpr.txt
    mv H.txt H_bpr.txt
    julia run_eval_bpr.jl
done
