# for i in {1..3}; do
#      taskset -c "51" python main.py --config "algorithms/mDSDI/configs/PACS_photo.json" --exp_idx $i --gpu_idx "1"
# done

# tensorboard --logdir "/data/habui/gradensity_inference/algorithms/ERM/results/tensorboards/MNIST_0"

# rm -r algorithms/mDSDI/results/checkpoints/*
# rm -r algorithms/mDSDI/results/logs/*
# rm -r algorithms/mDSDI/results/plots/*
# rm -r algorithms/mDSDI/results/tensorboards/*

python main.py --config "algorithms/ERM/configs/MNIST.json" --exp_idx "0" --gpu_idx "0"
# taskset -c "11" python main.py --config "algorithms/ERM/configs/Rotated_75_MNIST.json" --exp_idx "0" --gpu_idx "0"

python utils/tSNE_plot.py --plotdir "/home/ubuntu/gradensity_inference/algorithms/ERM/results/plots/MNIST_0/"
# python utils/tSNE_plot.py --plotdir "/home/ubuntu/gradensity_inference/algorithms/ERM/results/plots/Rotated_75_MNIST_0/" 

# tensorboard dev upload --logdir "/data/habui/gradensity_inference/algorithms/ERM/results/tensorboards/Rotated_75_MNIST_0"