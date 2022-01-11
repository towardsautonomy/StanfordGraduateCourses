#!/usr/bin/env bash
zip -r hw3.zip codebase/gan.py codebase/flow_network.py out*/fake_0900.png maf/samples_epoch50.png
echo "Submission created in hw3.zip"
