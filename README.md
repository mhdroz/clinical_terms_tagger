# clinical_terms_tagger

Code acccompanying the following manuscript: **Leveraging MedDRA Biomedical Terminology and Weak Labeling for Medical Concepts Extraction from CLinical Notes**

## Getting started


### Prerequisite
What to install and how to instal them
- Instructions to get the meddra terminology and which files are needed to build the tagger
- Instructions for where to find the base termino for WL
- Packages to install

```
Give examples
```

## How to run
Describe each step for a successful run:
1. tag the documents (include tagger creation if does not already exist)
3. weak labeling extractions
4. cleaning extractions

_include an example with a dummy termino and open source notes_

1. in the src/ directory, run the following command:
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



## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
