"""
I want to check if the n-th Chebyshev polynomials match the vector versions. I suspect not.
Or if they do, the summations have a roundoff error.
"""
from analysis.methods.chebyshev_vector import mps_chebyshev
from analysis.methods.vector import vector_chebyshev

mps = mps_chebyshev(10)
vector = vector_chebyshev(10)


# 3. Compare them in norm.
