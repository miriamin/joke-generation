# Computational Joke Generation

This repository contains two different joke generators, a scheme-based joke generator and a neural joke generator. Please consult the chapter "Implementation of joke generating algorithms" in the attached paper for more detailed documentation. 

## Scheme-based joke generator

The scheme-based joke generator is a reimplementation of reimplementation of Petrovic and Matthews (2013) as scheme-based generator and jokes that were generated using the following paper:

*Petrović, S., & Matthews, D. (2013). Unsupervised joke generation from big data.
  Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), 228–232.*

`pattern-joke-generator/generate_jokes.ipynb` is a notebook that allows to easily execute the generator and generate new jokes

You can only generate new jokes when you have the English News Corpus from 2016 (1 Million words) from Uni Leipzig's Deutscher Wortschatz projekt imported into a local MySQL database. Klick [here](http://wortschatz.uni-leipzig.de/de/download) to find the text collection and installation instructions.


## Neural joke generator

The neural joke generator generated jokes with Max Woolf's library [gpt-2-simple](https://github.com/minimaxir/gpt-2-simple). I fine-tuned the language model with [this](https://colab.research.google.com/drive/1VLG8e7YSEwypxU-noRNhsv5dW4NfTGce) Google Colaboratory Notebook template provided by Max Woolf. 


You can find the training data for the different fine-tuning scenarios in `neural-joke-generator/data/training-data` and the output in `neural-joke-generator/output`


## Evaluation

The notebook `evaluation/humorousness_evaluation.ipynb` contains the scripts for the evaluation of both generators.



