# Length Collapse
The implementation for ICLR 2025 Submission: Length-Induced Embedding Collapse in Transformer-based Models.

## Quick Start

- For details of testing framework, please check the code in the folder `src/test.py`.

- For details of Temperaute Scaling in model, please check the file `src/model.py`

## File Structure
```shell
.
├── src  # * datset for training and testing
│   ├── model.py # * Method TempScale 
│   └── test.py # * Testing Framework
```

## Quick Example

```python
python src/test.py --model ${model} --task ${task} --batch_size ${batch_size} --temperature ${temperature}
```

## Dependencies

This repository has the following dependency in requirements.txt.

```
beir==2.0.0
huggingface-hub==0.23.2
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.5.0
scipy==1.13.1
sentence-transformers==3.0.1
torch==2.3.0
```

The required packages can be installed via `pip install -r requirements.txt`.

