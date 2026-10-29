"""Exact BGE-M3 encoder, batched normalized vectors and content-addressed cache."""
import hashlib
from uuid import uuid4
import json
import time
from pathlib import Path
from functools import lru_cache

import numpy as np

from src.rag.settings import EMBEDDING_MODEL, RESULTS, setting


@lru_cache(maxsize=1)
def local_model():
    # Official ONNX export, CLS pooling as in BGE-M3 dense encoding. Disable
    # weight prepacking/graph rewrites to avoid a second multi-GB weight copy.
    import onnxruntime as ort
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer
    path=Path(snapshot_download(EMBEDDING_MODEL,cache_dir=str(RESULTS/'models'),
        revision=setting('BGE_MODEL_REVISION','main'),
        allow_patterns=['onnx/*','tokenizer.json','tokenizer_config.json','special_tokens_map.json','config.json']))
    options=ort.SessionOptions()
    options.intra_op_num_threads=2;options.inter_op_num_threads=1
    options.enable_cpu_mem_arena=False;options.enable_mem_pattern=False
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry('session.disable_prepacking','1')
    session=ort.InferenceSession(str(path/'onnx/model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    tokenizer=Tokenizer.from_file(str(path/'tokenizer.json'))
    tokenizer.enable_truncation(max_length=8192)
    tokenizer.enable_padding(pad_id=1,pad_token='<pad>')
    class OfficialONNX:
        def embed(self,texts,batch_size=1):
            for start in range(0,len(texts),batch_size):
                encoded=tokenizer.encode_batch(texts[start:start+batch_size])
                values={'input_ids':np.asarray([e.ids for e in encoded],dtype=np.int64),
                    'attention_mask':np.asarray([e.attention_mask for e in encoded],dtype=np.int64),
                    'token_type_ids':np.zeros((len(encoded),len(encoded[0].ids)),dtype=np.int64)}
                output=session.run(None,{i.name:values[i.name] for i in session.get_inputs()})[0]
                vectors=output[:,0,:] if output.ndim==3 else output
                yield from vectors
    return OfficialONNX()


class BGEEncoder:
    model_name=EMBEDDING_MODEL
    expected_dimension=1024

    def __init__(self,cache=None):
        self.cache=cache or RESULTS/'embedding_cache'
        self.cache.mkdir(parents=True,exist_ok=True)
        self.endpoint=setting('BGE_M3_ENDPOINT')

    def encode(self,texts,batch_size=4):
        if not texts:
            raise ValueError('At least one text is required.')
        signature=self.model_name+'|'+setting('BGE_MODEL_REVISION','main')+'|'+self.endpoint+'|normalized-8192-v1|'
        paths=[self.cache/(hashlib.sha256((signature+t).encode()).hexdigest()+'.npy') for t in texts]
        missing=[i for i,path in enumerate(paths) if not path.exists()]
        for start in range(0,len(missing),batch_size):
            indices=missing[start:start+batch_size]
            batch=[texts[i] for i in indices]
            if self.endpoint:
                import httpx
                key=setting('BGE_M3_API_KEY')
                headers={'Authorization':'Bearer '+key} if key else {}
                try:
                    response=httpx.post(self.endpoint,json={'model':self.model_name,'input':batch},headers=headers,timeout=120)
                    response.raise_for_status()
                    data=response.json()
                except Exception as exc:
                    raise RuntimeError('BGE-M3 embedding endpoint unavailable; no fallback encoder used.') from None
                if data.get('model') != self.model_name:
                    raise ValueError('Embedding endpoint must attest the exact BAAI/bge-m3 model.')
                vector=data.get('embeddings')
                if vector is None and 'data' in data:
                    vector=[item['embedding'] for item in sorted(data['data'],key=lambda item:item['index'])]
                vectors=np.asarray(vector,dtype=np.float32)
            else:
                vectors=np.asarray(list(local_model().embed(batch,batch_size=batch_size)),dtype=np.float32)
            if vectors.shape != (len(batch),self.expected_dimension) or not np.isfinite(vectors).all():
                raise ValueError('Actual BGE-M3 output has an invalid shape/dimension or non-finite values.')
            norms=np.linalg.norm(vectors,axis=1,keepdims=True)
            if (norms<=0).any():
                raise ValueError('Zero embedding vector is not valid.')
            vectors=vectors/norms
            for index,vector in zip(indices,vectors):
                path=paths[index]
                temporary=path.with_suffix('.'+uuid4().hex+'.tmp')
                with temporary.open('wb') as file:
                    np.save(file,vector,allow_pickle=False)
                temporary.replace(path)
        result=np.stack([np.load(path,allow_pickle=False) for path in paths])
        if result.shape != (len(texts),self.expected_dimension) or not np.isfinite(result).all() or (np.linalg.norm(result,axis=1)<=0).any():
            raise ValueError('Cached embedding dimension is incompatible.')
        return result
