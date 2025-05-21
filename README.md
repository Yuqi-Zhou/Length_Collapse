# Length Collapse
The implementation for ACL 2025: Length-Induced Embedding Collapse in PLM-based Models.

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

For example, you can set the "task=mteb/nfcorpus" to finish the retrieval task on NFCorpus dataset. All the datasets used in this paper can be found at https://huggingface.co/datasets/mteb. 

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

## Citation

If you find our code or work useful for your research, please cite our work.

```
@article{zhou2025length,
  title={Length-Induced Embedding Collapse in PLM-based Models},
  author={Zhou, Yuqi and Dai, Sunhao and Cao, Zhanshao and Zhang, Xiao and Xu, Jun},
  journal={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics},
  year={2025}
}
```
