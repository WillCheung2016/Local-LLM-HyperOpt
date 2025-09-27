
# Large Language Models for Hyperparameter Optimization

Michael R. Zhang, Nishkrit Desai, Juhan Bae, Jonathan Lorraine, Jimmy Ba

We explore the use of large language models (LLMs) in hyperparameter optimization (HPO). By prompting LLMs with dataset and model descriptions, we develop a methodology where LLMs suggest hyperparameter configurations, which are iteratively refined based on model performance. Our empirical evaluations on standard benchmarks reveal that LLMs, within constrained search budgets, can match or outperform traditional HPO methods like Bayesian optimization across different models on standard benchmarks. 

[Arxiv paper](https://arxiv.org/abs/2312.04528)


## Reproducing experiments on CIFAR

`cifar/train.py` implements our method on CIFAR-10 and can be used to reproduce the results in our paper. The provide code demonstrates how to conduct our experiments with Vision Transformers and a small ResNet. The same prompt is used for both models in our experiments to make the tuning task more difficult.

### Setup
1. Install [Ollama](https://ollama.com/download) on the machine running the experiments.
2. Ensure that the `ollama` CLI is available on your `PATH` (or set the
   `OLLAMA_EXECUTABLE` environment variable to point to the binary). The
   training scripts call the CLI directly so everything runs headlessly
   without needing to expose an HTTP server.
3. Pull the model you would like to use for hyper-parameter tuning, for example:

   ```
   ollama pull llama3
   ```

Install the Python dependencies:

```
pip install -r requirements.txt
```

### Usage

Run the script with desired arguments:
```
python train.py [-h] [--random_hparams | --llm] [--rounds ROUNDS] [--search_space SEARCH_SPACE]
                [--dataset_dir DATASET_DIR] [--arch ARCH] [--save_dir SAVE_DIR]
                [--train_batch_size TRAIN_BATCH_SIZE] [--eval_batch_size EVAL_BATCH_SIZE]
                [--learning_rate LEARNING_RATE] [--weight_decay WEIGHT_DECAY] [--label_smoothing LABEL_SMOOTHING]
                [--optimizer OPTIMIZER] [--num_train_epochs NUM_TRAIN_EPOCHS] [--seed SEED]
                [--max_train_batches MAX_TRAIN_BATCHES] [--max_eval_batches MAX_EVAL_BATCHES]
```

To run the LLM-based hyperparameter search with a single seed:
```
python train.py --llm --seed SEED
```
By default, this tunes the optimizer, learning rate, batch size, weight decay, and label smoothing. GPT-4 generated some ranges which we found reasonable. To reproduce our results, run the command with five different random seeds.

To run random hyperparameter search:
```
python train.py --random_hparams --seed SEED
```

This samples randomly from the same default configuration space. To reproduce our results, run the command with 100 different random seeds and then bootstrap to estimate the best error so far.

If neither --llm nor --random_hparams is specified, a training run is performed with hyperparameters specified by the other arguments.

### Quick CIFAR smoke test

To verify the end-to-end pipeline without committing to a full training run,
limit the number of batches processed during training and evaluation:

```
python -m cifar.train --dataset_dir ./data --save_dir ./out --arch resnet9 \
    --num_train_epochs 1 --train_batch_size 512 --eval_batch_size 1024 \
    --max_train_batches 2 --max_eval_batches 2
```

This command downloads CIFAR-10 as needed, trains for two batches, and evaluates
on two batches from both the train and validation splits. It exercises the full
data pipeline while keeping the runtime well under a minute on CPU-only setups.

### Trying different architectures

To try different architectures, modify the --arch argument when running the script. The currently supported architectures are:

resnet9: A small ResNet architecture with 9 layers

vit: A Vision Transformer (ViT) architecture

Example usage:
```
python train.py --arch resnet9
python train.py --arch vit
```

### Results 

The results of the hyperparameter search will be saved in the specified save_dir as JSON files. The files will contain the hyperparameters and corresponding evaluation metrics (loss and accuracy) for each trial.

### Adapting to other Datasets and Architectures
To provide more specific details on the dataset and architecture to potentially improve tuning performance, you can modify the initial_prompt in the main function of train.py.

For example, you can add information about the CIFAR-10 dataset, such as the number of classes, image dimensions, and any data augmentation techniques used. Additionally, you can provide more details about the chosen architecture, such as the number of layers, hidden dimensions, or attention heads for the ViT.

By providing more context about the dataset and architecture in the prompt, the LLM may be able to generate more informed hyperparameter suggestions.

## Toy experiments

You can find code we use to run 2D experiments in `run_toy_fns.ipynb`. 

## Cite 
If you find our work useful, please cite as:
```
@article{zhang2023using,
  title={Using Large Language Models for Hyperparameter Optimization},
  author={Zhang, Michael R and Desai, Nishkrit and Bae, Juhan and Lorraine, Jonathan and Ba, Jimmy},
  journal={arXiv preprint arXiv:2312.04528},
  year={2023}
}
```