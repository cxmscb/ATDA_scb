# ATDA
This repository contains code to reproduce results from the paper:

**Improving the Generalization of Adversarial Training with Domain Adaptation**

openreview report: https://openreview.net/forum?id=SyfIfnC5Ym

###### REQUIREMENTS

The code was tested with Python 3.6.5, Tensorflow 1.8.0, Keras 2.12, Keras_contrib 0.0.2,  Torchvision 0.2.1 and Numpy 1.14.3. 

###### EXPERIMENTS

We use  Adversarial Training  with Domain Adaptaion to train a main model (modelZ)  for Fashion-MNIST. These are described in *fashion_mnist.py*.

```
python -m train_atda models/modelZ_atda --type=0
```

Then, we use Normal Training to train a model (modelC) for  Fashion-MNIST.

```
python -m train models/modelC --type=3
```

To use Original/ Standard Adversarial Training to train a main model：

```
python -m train_adv models/modelZ_adv --type=0
```

To use Ensemble Adversarial Training to train a main model：

```
# First train pre-trained models:
python -m train models/modelA --type=1
python -m train models/modelB --type=2
# use Ensemble Adversarial Training method to train with pre-trained models
python -m train_adv models/modelZ_ens models/modelA models/modelB --type=0
```

The accuracy of the models on the Fashion MNIST test set can be computed using:

```
python -m simple_eval test [model(s)]
```

To evaluate robustness to various attacks, we use:

```
python -m simple_eval [attack] [source_model] [target_model(s)] [--parameters (opt)]
```

The attack can be:

| Attack | Description               | Parameters                                                   |
| ------ | ------------------------- | ------------------------------------------------------------ |
| fgs    | Standard FGSM             | *eps* (the norm of the perturbation)                         |
| rfgs   | RAND+FGSM                 | *eps* (the norm of the total perturbation); *alpha* (the norm of the random perturbation) |
| bim    | The iterative FGSM        | *eps* (the norm of the perturbation); *steps* (the number of iterative FGSM steps) |
| mim    | Momentum Iterative Method | The parameter is fixed in the function *momentum_fgs* of the *fgs.py*. |

###### competing methods

- Original / Standard Adversarial Training:  https://github.com/cxmscb/ensemble-adv-training
- Ensemble Adversarial Training：https://github.com/cxmscb/ensemble-adv-training
- Provably Robust Training：https://github.com/locuslab/convex_adversarial  (based pytorch)

Generalization  on adversarial white-box attacks for Fashion-MNIST ：

|                                             | FGSM Accuracy | BIM Accuracy | RAND+FGSM Accuracy | MI-FGSM Accuracy |
| ------------------------------------------- | ------------- | ------------ | ------------------ | ---------------- |
| Normal training                             | 8.3%          | 0.5%         | 15.0%              | 0.1%             |
| Original / Standard Adversarial Training    | 88.8%         | 13.9%        | 31.2%              | 9.4%             |
| Ensemble Adversarial Training               | 89.0%         | 12.8%        | 31.6%              | 6.6%             |
| Provably Robust Training                    | 67.4%         | 66.9%        | 72.2%              | 66.7%            |
| Adversarial Training with Domain Adaptation | 78.2%         | 70.2%        | 77.0%              | 68.8%            |

More information on Fashion-MNIST, SVHN, CIFAR010 and CIFAR-100 can be found in the [paper](https://openreview.net/forum?id=SyfIfnC5Ym)

###### Acknowledgments

Code refer heavily to:  [Ensemble Adversarial Training](https://github.com/cxmscb/ensemble-adv-training) 