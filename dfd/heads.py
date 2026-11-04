"""Factorized PARSeq-Ti output head interface (implementation pending).

Keep the encoder, decoder and composed-character context embeddings unchanged.
Three linear heads predict base/modifier/tone. Compose in log space and normalize
over the actual valid-character vocabulary, including an explicit EOS policy.
Reference: proposal sections 3.2–3.3; the head concept is inherited from VietTDR.
"""


class FactorizedHead:
    def __init__(self, hidden_size: int, vocabulary):
        raise NotImplementedError("Implement three torch linear heads and valid-character composition after PARSeq integration")
