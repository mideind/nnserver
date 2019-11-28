#!/bin/bash

gunicorn --bind 0.0.0.0:5000 nnserver:app --workers 3
