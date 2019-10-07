- each model will save its respective runs to saved/runs under the right folder
- be careful because the .pt weight files are huge and will fill up our dir space
- should try not to keep too many saved runs for the same model around consecutively


for the encoder/decoder model:
there is no padding required anymore in the dataset!!
the format of the dataset is TOKENIZED FIRST | TOKENIZED SECOND | TOKENIZED THIRD

the general approach is:
- take a batch of sentences (first + EOS + second) and encode them using ELMo
- ELMo will normalize and output a torch tensor of dim (batch, seq_max_len, 1024)
it pads all shorter sentences with 0s which is nice
- we use pack_padded_sequence like we did in hw3 to ignore the 0 padded vectors
and pass many variable-length sentences into the encoder GRU at the same time

our target is the ELMo embedding of the third sentence, so we pass these through in a batch too
- ELMo pads all of these again into a tensor

for the decoder, we predict one word PER-BATCH at each timestep and "unroll" the GRU
- problem is that not all 3rd sentences have the same length so we end up predicting extra
characters for the shorter ones
- cosine-embedding-loss takes in a vector (called label) in the code that specifies
for which pairs the loss is actually supposed to be calculated
- we essentially 0 out the loss for all "extra" predictions (but still calculate them computationally. sad)


things that are janky??
- our EOS character is just |
- when we predict elmo embeddings we predict the elmo embeddings from another elmo model
(not the one we're training as a part of our encoder) -- is this good/bad?
- the padding system is kinda weird 