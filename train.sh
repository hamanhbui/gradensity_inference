for i in {1..1}; do
    python main.py --config "algorithms/SM/configs/MNIST.json" --exp_idx $i --gpu_idx "1"
    # python main.py --config "algorithms/VAE/configs/MNIST.json" --exp_idx $i --gpu_idx "1"
    # python main.py --config "algorithms/CSM/configs/MNIST.json" --exp_idx $i --gpu_idx "1"
    # python main.py --config "algorithms/CVAE/configs/MNIST.json" --exp_idx $i --gpu_idx "1"
done

# taskset -c "51" python main.py --config "algorithms/mDSDI/configs/PACS_photo.json" --exp_idx $i --gpu_idx "1"
# python main.py --config "algorithms/CVAE/configs/MNIST.json" --exp_idx "1" --gpu_idx "0"
# python main.py --config "algorithms/ERM/configs/MNIST.json" --exp_idx "1" --gpu_idx "0"

# rm -r algorithms/SM/results/checkpoints/*
# rm -r algorithms/SM/results/logs/*
# rm -r algorithms/SM/results/plots/*
# rm -r algorithms/SM/results/tensorboards/*

# python utils/tSNE_plot.py --plotdir "/home/ubuntu/gradensity_inference/algorithms/CSM/results/plots/MNIST_1/"

# tensorboard --logdir "/data/habui/gradensity_inference/algorithms/SM/results/tensorboards/Rotated_75_MNIST_0"
# tensorboard dev upload --logdir "/data/habui/gradensity_inference/algorithms/SM/results/tensorboards/Rotated_75_MNIST_0"

# black -l 119 ./
# isort -l 119 --lai 2 -m 3 --up --sd 'FIRSTPARTY' -n --fgw 0 --tc ./