# ============================================================
# RAG Reproduction — Embeddings assigned
#
# Author: Andrew Bieber <andrewbieber.work@gmail.com>
# Last Updated: September 8, 2026
#
# File: embedder.py
#
# Description:
# Reads all chunks from each set of different chunk sizes
# and dumps the binary vectors into data/embeddings/*pt.
# We use the pretrained weights and BPE encoder of DPR given to
# us by the wonderful people who wrote this paper lol.
# YAY OPEN_SOURCE <3
#
# Paper:
# "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
# Lewis et al., 2020
# https://arxiv.org/abs/2005.11401
# ============================================================

