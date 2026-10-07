#!/bin/bash
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

cd /Users/maximilianfrohlich/Desktop/GitHub/yaqs/manuscript
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
