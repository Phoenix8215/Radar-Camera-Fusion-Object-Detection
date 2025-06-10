#!/bin/bash

workspace = "absolute/path/to/workspace"


trtexec --onnx=${workspace}/best.sim.onnx \
    --minShapes=images:1x3x640x640 \
    --maxShapes=images:4x3x640x640 \
    --optShapes=images:1x3x640x640 \
    --saveEngine=${workspace}/best.engine

