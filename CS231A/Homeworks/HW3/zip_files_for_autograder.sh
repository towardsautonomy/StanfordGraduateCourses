#!/usr/bin/env bash

cp p1/p1.py p1.py
cp p2/p2.py p2.py
cp p5/p5.py p5.py
zip -r submission.zip p1.py p2.py p5.py
rm p1.py
rm p2.py
rm p5.py
