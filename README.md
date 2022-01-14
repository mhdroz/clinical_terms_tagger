# clinical_terms_tagger

Code acccompanying the following manuscript: **Leveraging MedDRA Biomedical Terminology and Weak Labeling for Medical Concepts Extraction from CLinical Notes**

<br /><br />

![Pipeline](figures/pipeline.png)

<br /><br />

> __Usability note:__ This is experimental work, not a directly usable software library.
The code was developed in the context of an academic research project, highly
exploratory and iterative in nature. It is published here in the spirit of
open science and reproducibility values.

<br /><br />

## Getting started

### Prerequisite

The project was built using the Anaconda distribution of python 3.6.12. To run the code, clone this repository and use the Anaconda environment manager to create an environment using the `environment.yml` file to install the tagger code dependencies.

This code uses the [MedDRA](https://www.meddra.org) terminology as source for clinical concepts extraction. In order to use the terminology, a [subscritpion](https://www.meddra.org/how-to-use/support-documentation/english/welcome) is needed. Once access to the terminology files have been granted, concatenate all single level files into one terminology file for the creation of the tagger and save the terminology file in res/termino.txt with the following structure: `code|term|level`. 

The base terminology for the weak supervision part can be found there:

```
Give examples
```

## How to run
Describe each step for a successful run:
1. tag the documents (include tagger creation if does not already exist)
3. weak labeling extractions
4. cleaning extractions

_include an example with a dummy termino and open source notes_

1. Update the paths and file names in HP.py. In the src/ directory, run the following command:
```
nohup python meddra_extract.py --task 'tag' --label_model 'label_model' --threads 28 --datasource local > extract_meddra.out&
```
2. Once the tagging has finished, in the project directory, run the following command:
```
python setup_wl.py start_idx end_idx > wl.out&
```
3. Finally, to eliminate double occurrences of composite terms, in the same project directory, run:
```
python setup_cleaning.py start_idx end_idx n_threads > clean.out&
```
### Example with dummy data
An example with a dummy terminology file and notes is provided in example/


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
