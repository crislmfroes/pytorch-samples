#!/usr/bin/env bash
mkdir data
wget https://pytorch.org/tutorials/_static/img/neural-style/picasso.jpg
wget https://pytorch.org/tutorials/_static/img/neural-style/dancing.jpg
mv picasso.jpg data/
mv dancing.jpg data/