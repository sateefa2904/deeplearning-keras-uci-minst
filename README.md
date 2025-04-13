# Deep Learning with Keras on UCI + MNIST

Welcome to a neural playground of classic datasets and powerful models.  
This repo explores fully connected and convolutional neural networks using Keras — tested on MNIST and UCI datasets.

---

## What's Inside?

- `dense_mnist_base.py` – basic fully connected NN on MNIST  
- `cnn_mnist_base.py` – convolutional neural net on MNIST  
- `dense_mnist_opt.py` / `cnn_mnist_opt.py` – optimized variants  
- `nn_keras_solution.py` – final training + evaluation setup  
- `generate_test.py` – generate test splits  
- `uci_load.py` – loader for UCI datasets  
- `train.csv`, `test.csv` – input samples  
- `synthetic/`, `uci_datasets/` – data folders  
- `experiment_results.txt` – result logs for reference  

---

## Get Started

Install requirements:

```bash
pip install -r requirements.txt
```

Run any script (example):

```bash
python cnn_mnist_base.py
```

---

## Datasets Used

- [MNIST handwritten digits](http://yann.lecun.com/exdb/mnist/)
- UCI Datasets: Pendigits, Satellite, Yeast
- Synthetic datasets (`synthetic/`) for robustness checks

---

## Project Goals

- Compare dense vs. CNN architectures
- Analyze overfitting, dropout, and optimization
- Visualize and benchmark across datasets

---

## Author

Soli Ateefa – CS Honors @ UTA | Deep Learning Researcher  
Building models that decode the biological and digital world.

> Let's connect: [LinkedIn](https://linkedin.com/in/sateefa2904)

---

## License

MIT License. Free to use with credit
```