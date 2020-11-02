# TRP Detection on the Switchboard Corpus

This repo contains some of the files used in my dissertation. Many of the files are modified versions of those used by Roddy, Skantze, and Harte in the following repo (https://github.com/mattroddy/lstm_turn_taking_prediction).

As a **brief** guide to doing what I did, follow these instructions.

1. Download Switchboard Audio and corresponding NXT Corpus. Then run _sph2wav.sh_ and _split_channels.sh_ to separate the audio into individual speakers and a wav format.
2. Use Roddy's feature extraction pipeline to get acoustic features for each audio file.
3. To get lexical features, use my versions of _get_word_timings.py_ (instead of Roddy's _get_VA_annotations.py_), _get_vocab.py_, and _get_word_annotations.py_. The other data prep files in Roddy's pipeline only require minor adaptation.
4. Extract the TRPs using _get_dialogue_acts.py_
5. Train an LSTM using Roddy's script (icmi...)
6. Get the frames you will use to train the feedforward model using _trp_training_data_idx.py_, running once for training data, and once for dev data.
7. Train the feedforward network running _train_feedforward_redux.py_
8. Get TRPs using _find_trps.py_
9. Evaluate TRPs using _trp_pk.py_
