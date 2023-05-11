# SimpleUpdate
A julia package implementing SimpleUpdate for graph-PEPS and graph-PESS tensor networks

**Dependencies**
- TensorOperations
- StaticArrays
- Combinatorics


**Modules**
- Util: Mostly common code for PEPS and PESS
- Operator: Operator type
- OptimalContraction: Using TensorOperations optimalcontraction to build a cache
for optimal contractions
- gPEPS: graph-PEPS simple update algorithm
- gPESS: graph-PESS simple update algorithm
