# ReLIE: Representation-Learning-for-Information-Extraction
This is an unofficial implementation of Representation Learning for Information Extraction ([ReLIE](https://research.google/pubs/pub49122/)) from Form-like Documents using PyTorch.

## Model Architecture

![Architecture](./assets/images/scoring_network.png)

[image source](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/59f3bb33216eae711b36f3d8b3ee3cc67058803f.pdf)

## Getting Started

1. Clone the repository
```
git clone https://github.com/Praneet9/Representation-Learning-for-Information-Extraction.git
```
2. Create a virtualenv and install the required packages
```
pip install -r requirements.txt
```

## Train
### Prepare dataset  
* STEP 1: Generate the tesseract results for the invoices using [generate_tesseract_results](utils/generate_tesseract_results.py)
* STEP 2: Extract the candidates using [extract_candidates.py](utils/extract_candidates.py)
* STEP 3: Use the labelled annotations to get and attach neighbors to the candidates using [xml_parser.py](utils/xml_parser.py)(Ex. Pascal VOC) and [Neighbour.py](utils/Neighbour.py)
* STEP 4: Normalize the Cartesian coordinates of each neighbors using [operations.py](utils/operations.py)
* STEP 5: Preprocess the inputs using [preprocess.py](utils/preprocess.py)
* STEP 6: Run [train.py](train.py)

## Inference
* Get the inference results by running
```
python3 inference.py --image sample.jpg --cuda --cached_pickle output/cached_data.pickle --load_saved_model output/model.pth
```

## Citation

##### Representation Learning for Information Extraction from Form-like Documents
_Bodhisattwa Prasad Majumder†♣ Navneet Potti♠ Sandeep Tata♠
James B. Wendt♠ Qi Zhao♠ Marc Najork♠_ <br>

**Abstract** <br>
We propose a novel approach using representation learning for tackling the problem of extracting structured information from form-like
document images. We propose an extraction
system that uses knowledge of the types of the
target fields to generate extraction candidates,
and a neural network architecture that learns a
dense representation of each candidate based
on neighboring words in the document. These
learned representations are not only useful in
solving the extraction task for unseen document templates from two different domains,
but are also interpretable, as we show using
loss cases.

[[Paper]](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/59f3bb33216eae711b36f3d8b3ee3cc67058803f.pdf) [[Google Blog]](https://ai.googleblog.com/2020/06/extracting-structured-data-from.html) 

```
@article{
  title={Representation Learning for Information Extraction from Form-like Documents},
  author={Bodhisattwa Prasad Majumder, Navneet Potti, Sandeep Tata, James B. Wendt, Qi Zhao, Marc Najork},
  journal = {Association for Computational Linguistics},
  year={2020}
}
```