# ReLIE: Representation-Learning-for-Information-Extraction
This is an implementation of [ReLIE](https://research.google/pubs/pub49122/) for extracting information from invoices on PyTorch.

## Model Architecture

![Architecture](./assets/images/scoring_network.png)
[Image taken from the mentioned paper](https://research.google/pubs/pub49122)

## Getting Started

1. Clone the repository
```
git clone https://github.com/vickipedia6/Generating_TV_Scripts.git
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
